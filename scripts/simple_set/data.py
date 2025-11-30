# scripts/simple_set/data.py

import torch
from torch.utils.data import Dataset
import logging
from scripts.autoencoder.imdb_row_autoencoders import TitlesAutoencoder
from scripts.autoencoder.row_autoencoder import _apply_field_state

class HybridSetDataset(Dataset):
    def __init__(self, cache_path: str, cfg):
        super().__init__()
        logging.info(f"Loading hybrid dataset from {cache_path}...")
        data = torch.load(cache_path, map_location="cpu")
        
        self.stacked_fields = data["stacked_fields"] 
        # heads_ragged[head] = {'flat': T, 'offsets': T, 'lengths': T}
        self.heads_ragged = data["heads_ragged"] 
        
        self.num_people = data["num_people"]
        self.idx_to_name = data["idx_to_person_name"]
        
        temp_ae = TitlesAutoencoder(cfg)
        self.fields = temp_ae.fields
        
        field_configs = data["field_configs"]
        for f in self.fields:
            if f.name in field_configs:
                _apply_field_state(f, field_configs[f.name])
        
        self.num_items = self.stacked_fields[0].shape[0]
        
    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        return idx

def collate_hybrid_set(batch_indices, dataset):
    """
    Optimized Collate:
    1. Slices inputs (fast).
    2. Slices ragged target arrays to get indices (fast).
    3. Returns coordinates (indices) for sparse construction on GPU.
    """
    # 1. Inputs
    indices_t = torch.tensor(batch_indices, dtype=torch.long)
    batch_inputs = [t[indices_t] for t in dataset.stacked_fields]
    
    # 2. Targets (Sparse Coordinates)
    # We want to return:
    #   coords_dict[head] -> Tensor(N_total_targets, 2) where cols are [batch_idx, person_idx]
    #   counts_dict[head] -> Tensor(B, 1)
    
    coords_dict = {}
    counts_dict = {}
    
    for head, ragged in dataset.heads_ragged.items():
        flat = ragged['flat']
        offsets = ragged['offsets']
        lengths = ragged['lengths']
        
        # A. Counts (Easy slicing)
        # lengths is (NumMovies,), we slice (B,)
        batch_lengths = lengths[indices_t]
        counts_dict[head] = batch_lengths.float().unsqueeze(1)
        
        # B. Indices (Complex slicing optimized)
        # We need to extract the segments for each batch item from `flat`.
        # Since the batch_indices might be random (shuffled), the segments are scattered.
        
        # Get start/end for each item in batch
        starts = offsets[indices_t]
        ends = offsets[indices_t + 1]
        
        # We create a mask or index selection.
        # Since PyTorch doesn't have a vectorized "slice multiple ranges" easily without padding,
        # we can use a loop in Python (fast enough for batch size 512) or 
        # use repeat_interleave if we want pure torch.
        
        # repeat_interleave approach:
        # Create row indices: [0, 0, 0, 1, 2, 2...]
        row_indices = torch.repeat_interleave(
            torch.arange(len(batch_indices), dtype=torch.long), 
            batch_lengths
        )
        
        if row_indices.numel() > 0:
            # Construct the gather indices for the 'flat' tensor.
            # We need to generate [start[0]...end[0], start[1]...end[1], ...]
            # This logic is tricky to vectorize fully efficiently.
            # A semi-vectorized approach:
            
            # 1. Create a base ranges tensor
            # This is often the bottleneck. 
            # Fastest Python-loop approach usually wins for Ragged tensors in Dataloaders:
            
            gathered_values = []
            for i, (s, e) in enumerate(zip(starts.tolist(), ends.tolist())):
                if e > s:
                    gathered_values.append(flat[s:e])
            
            if gathered_values:
                col_indices = torch.cat(gathered_values)
                # Ensure row_indices matches col_indices length (it should by definition)
                assert row_indices.size(0) == col_indices.size(0)
                
                # Stack: (N, 2) -> [row, col]
                coords = torch.stack([row_indices, col_indices], dim=1)
                coords_dict[head] = coords
            else:
                coords_dict[head] = torch.empty((0, 2), dtype=torch.long)
        else:
            coords_dict[head] = torch.empty((0, 2), dtype=torch.long)

    return batch_inputs, coords_dict, counts_dict, indices_t