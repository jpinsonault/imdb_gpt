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
        # heads_padded[head] = Tensor(NumMovies, MaxLen) filled with -1
        self.heads_padded = data["heads_padded"] 
        
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
    Optimized Collate using Padded Tensors:
    1. Slices inputs (fast).
    2. Slices padded target arrays (fast).
    3. Uses masked selection to generate sparse coords without Python loops.
    """
    # 1. Inputs
    indices_t = torch.tensor(batch_indices, dtype=torch.long)
    batch_inputs = [t[indices_t] for t in dataset.stacked_fields]
    
    # 2. Targets (Sparse Coordinates)
    # Return:
    #   coords_dict[head] -> Tensor(N_total_targets, 2) [batch_idx, person_idx]
    #   counts_dict[head] -> Tensor(B, 1)
    
    coords_dict = {}
    counts_dict = {}
    
    for head, padded_tensor in dataset.heads_padded.items():
        # Slice the batch: (B, MaxLen)
        batch_padded = padded_tensor[indices_t]
        
        # Create mask for valid entries (not -1)
        mask = (batch_padded != -1)
        
        # A. Counts: Simply sum the mask
        counts = mask.sum(dim=1, keepdim=True).float()
        counts_dict[head] = counts
        
        # B. Indices: Vectorized coordinate generation
        # torch.nonzero returns indices where mask is True.
        # Since we want (batch_idx, person_idx), we construct it manually 
        # to ensure we get the values from batch_padded.
        
        # This gets us the indices (row_in_batch, col_in_padded)
        nonzero_indices = torch.nonzero(mask, as_tuple=True)
        row_indices = nonzero_indices[0] 
        col_indices = nonzero_indices[1]
        
        # Extract actual person IDs using the mask/indices
        person_ids = batch_padded[row_indices, col_indices]
        
        # We generally want coords to be int64 (Long) for downstream embedding lookups
        if person_ids.numel() > 0:
            coords = torch.stack([row_indices, person_ids.long()], dim=1)
            coords_dict[head] = coords
        else:
            coords_dict[head] = torch.empty((0, 2), dtype=torch.long)

    return batch_inputs, coords_dict, counts_dict, indices_t