from __future__ import annotations
import logging
import sqlite3
import numpy as np
from typing import List, Tuple, Dict, Optional
import torch
from torch.utils.data import Dataset


# The functions 'WeightedEdgeSampler' and 'make_edge_sampler' are removed.

class EdgeEpochDataset(Dataset):
    """
    A simple wrapper around the pre-computed tensor cache for joint autoencoder.
    Used for a simple epoch-based training loop with built-in shuffling/data loading.
    This replaces all functionality of the previous dynamic sampling logic.
    """
    def __init__(self, cache_path: str):
        super().__init__()
        data = torch.load(cache_path, map_location="cpu")
        self.edge_ids = data["edge_ids"].long()
        self.movie_tensors = data["movie"]
        self.person_tensors = data["person"]

    def __len__(self):
        return int(self.edge_ids.shape[0])

    def __getitem__(self, idx: int):
        i = int(idx)
        # Note: Datasets generally return tuples/lists, which DataLoader will convert to tensors/batches.
        # We assume the movie/person tensors are lists of tensors (one per field).
        m = tuple(t[i] for t in self.movie_tensors)
        p = tuple(t[i] for t in self.person_tensors)
        eid = int(self.edge_ids[i])
        return m, p, eid
