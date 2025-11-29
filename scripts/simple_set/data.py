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
        
        self.stacked_fields = data["stacked_fields"] # List[Tensor(N, ...)]
        self.target_indices = data["target_indices"] # List[List[int]]
        self.num_people = data["num_people"]
        self.idx_to_name = data["idx_to_person_name"]
        
        # Reconstruct Field Objects for Recon/Model Init
        # We need an instance of TitlesAutoencoder to get the class definitions
        temp_ae = TitlesAutoencoder(cfg)
        self.fields = temp_ae.fields
        
        # Apply saved state (vocab, stats) to fields
        field_configs = data["field_configs"]
        for f in self.fields:
            if f.name in field_configs:
                _apply_field_state(f, field_configs[f.name])
        
        self.num_items = self.stacked_fields[0].shape[0]
        
    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        # We return the index, collate will slice the big tensors
        # This avoids copying tensors in individual workers
        return idx

def collate_hybrid_set(batch_indices, dataset):
    """
    batch_indices: List[int]
    """
    # 1. Slice Input Tensors
    # We use the dataset reference to access the big tensors directly
    # This is fast because we are just slicing memory
    indices = torch.tensor(batch_indices, dtype=torch.long)
    
    batch_inputs = []
    for big_tensor in dataset.stacked_fields:
        batch_inputs.append(big_tensor[indices])
        
    # 2. Build Multi-Hot Targets
    B = len(batch_indices)
    multi_hot_targets = torch.zeros(B, dataset.num_people, dtype=torch.float32)
    count_targets = torch.zeros(B, 1, dtype=torch.float32)
    
    for i, idx in enumerate(batch_indices):
        p_idxs = dataset.target_indices[idx]
        if p_idxs:
            t_idxs = torch.tensor(p_idxs, dtype=torch.long)
            multi_hot_targets[i, t_idxs] = 1.0
            count_targets[i] = float(len(p_idxs))
            
    return batch_inputs, multi_hot_targets, count_targets, indices