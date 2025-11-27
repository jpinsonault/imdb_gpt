# scripts/set_decoder/data.py

import torch
from torch.utils.data import Dataset
from typing import List

class CachedSetDataset(Dataset):
    def __init__(self, cache_path: str, num_slots: int):
        super().__init__()
        data = torch.load(cache_path, map_location="cpu")
        
        self.movie_latents = data["movie_latents"]
        self.person_latents = data["person_latents"]
        self.person_targets = data["person_targets"]
        self.indices = data["indices"]
        self.masks = data["masks"]
        self.movies = data.get("tconsts", []) 
        
        self.num_slots = int(num_slots)
        
        cached_slots = self.indices.shape[1]
        if cached_slots < self.num_slots:
            raise ValueError(f"Cache built with {cached_slots} slots, but config requested {self.num_slots}. Please refresh cache.")

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, idx):
        # 1. Movie Latent
        z_movie = self.movie_latents[idx]
        
        # 2. Get indices for this movie
        idxs = self.indices[idx, :self.num_slots]
        mask = self.masks[idx, :self.num_slots].float()
        
        # 3. Gather Person Latents
        # Replace -1 with 0 for gathering, then mask out result
        valid_idxs = idxs.clone()
        valid_idxs[idxs == -1] = 0
        
        Z_gt = self.person_latents[valid_idxs] # (Slots, Latent)
        Z_gt = Z_gt * mask.unsqueeze(-1)
        
        # 4. Gather Person Targets
        Y_fields = []
        for tgt_tensor in self.person_targets:
            Y_sample = tgt_tensor[valid_idxs]
            # Reshape mask to broadcast: (Slots, 1, 1...)
            view_shape = [-1] + [1] * (Y_sample.dim() - 1)
            Y_sample = Y_sample * mask.view(*view_shape)
            Y_fields.append(Y_sample)

        return z_movie, Z_gt, mask, Y_fields


def collate_set_decoder(batch):
    z_movies, Z_gts, masks, Y_fields_list = zip(*batch)

    z_movies = torch.stack(z_movies, dim=0)
    Z_gts = torch.stack(Z_gts, dim=0)
    masks = torch.stack(masks, dim=0)

    num_fields = len(Y_fields_list[0])
    Y_batch: List[torch.Tensor] = []
    for fi in range(num_fields):
        ys = [sample[fi] for sample in Y_fields_list]
        Y_batch.append(torch.stack(ys, dim=0))

    return z_movies, Z_gts, masks, Y_batch