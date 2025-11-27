# scripts/set_decoder/training.py

import sqlite3
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from scripts.autoencoder.fields import (
    ScalarField,
    BooleanField,
    MultiCategoryField,
    SingleCategoryField,
    TextField,
    NumericDigitCategoryField,
)
from scripts.set_decoder.model import SetDecoder


def _hungarian(cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Solves linear sum assignment using Scipy (much faster than manual Python loop).
    Input: (N, M) cost matrix on GPU/CPU
    Output: (Rows, Cols) indices on same device as input
    """
    # Move to CPU for scipy
    c_cpu = cost.detach().cpu().numpy()
    
    # Scipy handles rectangular matrices automatically
    row_ind, col_ind = linear_sum_assignment(c_cpu)
    
    return (
        torch.tensor(row_ind, dtype=torch.long, device=cost.device),
        torch.tensor(col_ind, dtype=torch.long, device=cost.device),
    )


def _field_loss_broadcast(field, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """
    Computes loss between every prediction slot and every target.
    pred: (B, N, ...)
    tgt:  (B, K, ...)
    Returns: (B, N, K)
    """
    # Expand dimensions for broadcasting:
    # pred -> (B, N, 1, ...)
    # tgt  -> (B, 1, K, ...)
    
    if isinstance(field, TextField):
        # pred: (B, N, L, V)
        # tgt:  (B, K, L)
        B, N, L, V = pred.shape
        K = tgt.shape[1]
        
        # We need to compute cross entropy for every pair.
        # Reshape to flattened batch for F.cross_entropy
        # pred: (B, N, 1, L, V) -> expand -> (B, N, K, L, V)
        pred_exp = pred.unsqueeze(2).expand(-1, -1, K, -1, -1)
        # tgt:  (B, 1, K, L)    -> expand -> (B, N, K, L)
        tgt_exp = tgt.unsqueeze(1).expand(-1, N, -1, -1).long()
        
        # Flatten to: (B*N*K*L, V) and (B*N*K*L)
        flat_pred = pred_exp.reshape(B * N * K * L, V)
        flat_tgt = tgt_exp.reshape(B * N * K * L)
        
        pad_id = int(getattr(field, "pad_token_id", 0) or 0)
        
        loss = F.cross_entropy(flat_pred, flat_tgt, ignore_index=pad_id, reduction="none")
        
        # Reshape back to (B, N, K, L)
        loss = loss.view(B, N, K, L)
        
        # Mask padding
        mask = (tgt_exp != pad_id).float()
        denom = mask.sum(dim=-1).clamp_min(1.0)
        
        # Sum over sequence length L
        return (loss * mask).sum(dim=-1) / denom

    if isinstance(field, NumericDigitCategoryField):
        # pred: (B, N, P, V)
        # tgt:  (B, K, P)
        B, N, P, V = pred.shape
        K = tgt.shape[1]
        
        pred_exp = pred.unsqueeze(2).expand(-1, -1, K, -1, -1)
        tgt_exp = tgt.unsqueeze(1).expand(-1, N, -1, -1).long()
        
        mask_id = int(field.mask_index) if field.mask_index is not None else -1000
        
        loss = F.cross_entropy(
            pred_exp.reshape(-1, V),
            tgt_exp.reshape(-1),
            ignore_index=mask_id,
            reduction="none"
        )
        loss = loss.view(B, N, K, P)
        return loss.mean(dim=-1)

    if isinstance(field, ScalarField):
        # pred: (B, N, 1)
        # tgt:  (B, K, 1)
        diff = pred.unsqueeze(2) - tgt.unsqueeze(1) # (B, N, K, 1)
        return diff.pow(2).mean(dim=-1)

    if isinstance(field, BooleanField):
        # pred: (B, N, 1)
        # tgt:  (B, K, 1)
        if getattr(field, "use_bce_loss", True):
            p_exp = pred.unsqueeze(2).expand(-1, -1, tgt.size(1), -1)
            t_exp = tgt.unsqueeze(1).expand(-1, pred.size(1), -1, -1)
            loss = F.binary_cross_entropy_with_logits(p_exp, t_exp, reduction="none")
            return loss.mean(dim=-1)
        else:
            diff = torch.tanh(pred.unsqueeze(2)) - tgt.unsqueeze(1)
            return diff.pow(2).mean(dim=-1)

    if isinstance(field, MultiCategoryField):
        # pred: (B, N, C)
        # tgt:  (B, K, C)
        p_exp = pred.unsqueeze(2).expand(-1, -1, tgt.size(1), -1)
        t_exp = tgt.unsqueeze(1).expand(-1, pred.size(1), -1, -1)
        loss = F.binary_cross_entropy_with_logits(p_exp, t_exp, reduction="none")
        return loss.mean(dim=-1)

    if isinstance(field, SingleCategoryField):
        # pred: (B, N, C)
        # tgt:  (B, K, 1)
        B, N, C = pred.shape
        K = tgt.shape[1]
        
        p_exp = pred.unsqueeze(2).expand(-1, -1, K, -1) # (B, N, K, C)
        t_exp = tgt.unsqueeze(1).expand(-1, N, -1, -1).long().squeeze(-1) # (B, N, K)
        
        loss = F.cross_entropy(
            p_exp.reshape(-1, C),
            t_exp.reshape(-1),
            reduction="none"
        )
        return loss.view(B, N, K)

    # Fallback
    diff = pred.unsqueeze(2) - tgt.unsqueeze(1)
    return diff.pow(2).mean(dim=-1)


def _compute_cost_matrices(
    people_ae,
    z_slots: torch.Tensor,
    Z_gt: torch.Tensor,
    Y_gt_fields: List[torch.Tensor],
    mask: torch.Tensor,
    w_latent: float,
    w_recon: float,
):
    """
    Vectorized computation of cost matrices.
    Returns List of (N, K_b) matrices because K_b varies per batch item.
    """
    B, N, D = z_slots.shape
    
    # We assume Z_gt is (B, MaxK, D)
    # However, the dataset collation might stack them densely.
    # Let's check shapes. If Z_gt comes from collate_set_decoder, it is (B, MaxK, D).
    # But wait, the CachedSetDataset logic for targets might have variable padding?
    # Yes, data.py logic: Z_gt = Z_gt * mask.unsqueeze(-1).
    # So Z_gt is (B, Slots, Latent). 
    
    # We need valid targets only. 
    # Optimization: We will compute the dense (B, N, MaxK) cost matrix first,
    # then slice it for every batch item.
    
    MaxK = Z_gt.shape[1]
    
    dec_device = next(people_ae.decoder.parameters()).device
    slot_device = z_slots.device

    # 1. Latent Cost (B, N, MaxK)
    # z_slots: (B, N, D) -> (B, N, 1, D)
    # Z_gt:    (B, MaxK, D) -> (B, 1, MaxK, D)
    diff_lat = z_slots.unsqueeze(2) - Z_gt.unsqueeze(1)
    C_lat_batch = diff_lat.pow(2).sum(dim=-1) # (B, N, MaxK)

    # 2. Recon Cost (B, N, MaxK)
    # Run decoder on all slots
    z_flat = z_slots.reshape(B * N, D).to(dec_device)
    dec_out = people_ae.decoder(z_flat) # List of tensors

    C_rec_batch = torch.zeros((B, N, MaxK), device=dec_device)

    # Y_gt_fields are list of (B, MaxK, ...)
    # dec_out is list of (B*N, ...)
    
    for fi, field in enumerate(people_ae.fields):
        pred_f = dec_out[fi] # (B*N, ...)
        
        # Reshape pred back to (B, N, ...)
        shape_suffix = pred_f.shape[1:]
        pred_f = pred_f.view(B, N, *shape_suffix)
        
        tgt_f = Y_gt_fields[fi].to(dec_device) # (B, MaxK, ...)
        
        # Vectorized loss for this field
        # Returns (B, N, MaxK)
        loss_f = _field_loss_broadcast(field, pred_f, tgt_f)
        
        C_rec_batch += loss_f

    # 3. Combine and Split
    # We need to return lists because Hungarian matcher runs per-sample with different K
    
    C_lat_batch = C_lat_batch.to(dec_device)
    C_total_batch = (w_latent * C_lat_batch + w_recon * C_rec_batch).detach()
    
    C_match_list = []
    C_lat_list = []
    C_rec_list = []
    
    # Pre-calculate K for each batch to avoid CPU sync in loop if possible
    # But mask summation is fast enough usually.
    
    for b in range(B):
        k_b = int(mask[b].sum().item())
        if k_b == 0:
            C_match_list.append(None)
            C_lat_list.append(None)
            C_rec_list.append(None)
            continue
            
        # Slice valid targets (0 to k_b)
        # Rows (N) are always full size
        c_match = C_total_batch[b, :, :k_b]
        c_lat = C_lat_batch[b, :, :k_b]
        c_rec = C_rec_batch[b, :, :k_b]
        
        C_match_list.append(c_match)
        C_lat_list.append(c_lat.to(slot_device))
        C_rec_list.append(c_rec.to(slot_device))

    return C_match_list, C_lat_list, C_rec_list

