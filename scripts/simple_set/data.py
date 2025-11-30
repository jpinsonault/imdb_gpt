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
        
        # Stacked fields: List[Tensor]
        self.stacked_fields = data["stacked_fields"] 
        # heads_padded[head] = Tensor(NumMovies, MaxLen) filled with -1
        self.heads_padded = data["heads_padded"] 
        
        # Pin memory for faster GPU transfer
        if torch.cuda.is_available():
            logging.info("Pinning dataset memory...")
            self.stacked_fields = [t.pin_memory() for t in self.stacked_fields]
            self.heads_padded = {k: v.pin_memory() for k, v in self.heads_padded.items()}
        
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


class FastInfiniteLoader:
    """
    A lightweight iterator that replaces DataLoader for in-memory tensor datasets.
    It slices the tensors directly on the main thread, avoiding multiprocessing overheads.
    """
    def __init__(self, dataset: HybridSetDataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.indices = torch.arange(len(dataset))
        if self.shuffle:
            self.indices = self.indices[torch.randperm(len(dataset))]
            
        self.ptr = 0
        self.n = len(dataset)

    def __iter__(self):
        return self

    def __next__(self):
        if self.ptr + self.batch_size > self.n:
            if self.shuffle:
                self.indices = self.indices[torch.randperm(self.n)]
            self.ptr = 0
            
        batch_idx = self.indices[self.ptr : self.ptr + self.batch_size]
        self.ptr += self.batch_size
        
        # Slice Inputs (List of Tensors)
        inputs = [t[batch_idx] for t in self.dataset.stacked_fields]
        
        # Slice Heads (Dict of Tensors)
        heads_padded_batch = {k: v[batch_idx] for k, v in self.dataset.heads_padded.items()}
        
        return inputs, heads_padded_batch, batch_idx

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

def collate_hybrid_set(batch_indices, dataset):
    # Compatibility shim if needed
    pass