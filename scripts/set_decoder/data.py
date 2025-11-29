# scripts/set_decoder/data.py

import torch
from torch.utils.data import Dataset
from typing import List, Tuple

class CachedSequenceDataset(Dataset):
    def __init__(self, cache_path: str, max_len: int):
        super().__init__()
        data = torch.load(cache_path, map_location="cpu")
        
        # Lists of tensors (one tensor per field)
        self.movie_inputs = data["movie_inputs"] 
        self.person_inputs = data["person_inputs"]
        self.person_targets = data["person_targets"]
        
        self.indices = data["indices"]
        self.masks = data["masks"]
        self.movies = data.get("tconsts", []) 
        
        self.max_len = int(max_len)
        
        cached_len = self.indices.shape[1]
        if cached_len < self.max_len:
            raise ValueError(f"Cache built with length {cached_len}, but config requested {self.max_len}. Please refresh cache.")

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, idx):
        # 1. Gather Movie Field Inputs
        # self.movie_inputs is a list of [Tensor(N, ...), Tensor(N, ...)]
        m_fields = []
        for field_tensor in self.movie_inputs:
            m_fields.append(field_tensor[idx])

        # 2. Get indices for this sequence
        idxs = self.indices[idx, :self.max_len]
        mask = self.masks[idx, :self.max_len].float()
        
        # 3. Gather Person Field Inputs (Sequence)
        # Handle -1 indices (padding) by mapping to 0 temporarily, then masking
        valid_idxs = idxs.clone()
        valid_idxs[idxs == -1] = 0
        
        p_fields_input = []
        for field_tensor in self.person_inputs:
            # Gather: (SeqLen, ...)
            gathered = field_tensor[valid_idxs]
            
            # If the field tensor has extra dims, unsqueeze mask
            # mask is (SeqLen)
            view_shape = [-1] + [1] * (gathered.dim() - 1)
            
            # If field is integer (e.g. text/categorical), we multiply by mask (0 becomes pad)
            # If float, multiply by mask
            # But wait, integer fields usually have 0 as pad or special.
            # BaseField handles padding logic, but here we are in raw tensor land.
            # Simple approach: multiply floats, keep ints as is? 
            # Actually, standard gather is enough. The 'mask' tensor is passed to the trainer 
            # to zero out gradients/losses for invalid steps.
            # We just need to make sure the inputs don't crash the encoder.
            p_fields_input.append(gathered)

        # 4. Gather Person Targets (for Recon Loss)
        p_fields_target = []
        for field_tensor in self.person_targets:
            gathered = field_tensor[valid_idxs]
            p_fields_target.append(gathered)

        return m_fields, p_fields_input, p_fields_target, mask


def collate_seq_decoder(batch):
    # batch is list of tuples (m_fields, p_fields_in, p_fields_tgt, mask)
    m_list, p_in_list, p_tgt_list, mask_list = zip(*batch)
    
    # 1. Collate Movie Fields: List[List[Tensor]] -> List[Tensor(Batch, ...)]
    num_m_fields = len(m_list[0])
    M_batch = []
    for i in range(num_m_fields):
        # stack the i-th field across batch
        M_batch.append(torch.stack([item[i] for item in m_list], dim=0))
        
    # 2. Collate Person Input Fields: List[List[Tensor(Seq, ...)]] -> List[Tensor(Batch, Seq, ...)]
    num_p_fields = len(p_in_list[0])
    P_in_batch = []
    for i in range(num_p_fields):
        P_in_batch.append(torch.stack([item[i] for item in p_in_list], dim=0))

    # 3. Collate Person Target Fields
    P_tgt_batch = []
    for i in range(num_p_fields):
        P_tgt_batch.append(torch.stack([item[i] for item in p_tgt_list], dim=0))

    # 4. Masks
    masks = torch.stack(mask_list, dim=0)

    return M_batch, P_in_batch, P_tgt_batch, masks