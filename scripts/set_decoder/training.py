# scripts/set_decoder/training.py

import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple

from scripts.autoencoder.fields import (
    ScalarField,
    BooleanField,
    MultiCategoryField,
    SingleCategoryField,
    TextField,
    NumericDigitCategoryField,
)

def match_batch_hungarian(
    cost_matrix_batch: torch.Tensor, 
    lengths: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs Hungarian matching on a batch of cost matrices on CPU.
    """
    # Move to CPU / Numpy for Scipy
    C_cpu = cost_matrix_batch.detach().cpu().numpy()
    lengths_cpu = lengths.cpu().numpy()
    
    b_indices = []
    r_indices = []
    c_indices = []
    
    B, N, _ = C_cpu.shape
    
    for i in range(B):
        k = int(lengths_cpu[i])
        if k == 0:
            continue
            
        # Slice valid submatrix (N, k)
        cost_sub = C_cpu[i, :, :k]
        
        # Scipy linear_sum_assignment
        # Replace NaNs to prevent crashes if training is unstable
        cost_sub = np.nan_to_num(cost_sub, nan=1e9, posinf=1e9, neginf=1e9)
        
        rows, cols = linear_sum_assignment(cost_sub)
        
        # Accumulate indices
        b_indices.extend([i] * len(rows))
        r_indices.extend(rows)
        c_indices.extend(cols)
        
    device = cost_matrix_batch.device
    return (
        torch.tensor(b_indices, dtype=torch.long, device=device),
        torch.tensor(r_indices, dtype=torch.long, device=device),
        torch.tensor(c_indices, dtype=torch.long, device=device)
    )

def _field_loss_broadcast(field, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """
    Computes loss matrix (B, N, K).
    Optimized to avoid creating (B, N, K, L, V) tensors which OOM.
    """
    if isinstance(field, TextField):
        B, N, L, V = pred.shape
        K = tgt.shape[1] # tgt is (B, K, L)

        # Reshape for optimized cross entropy
        # (B, N, 1, L, V) vs (B, 1, K, L)
        
        # We manually expand only the dims necessary to broadcast into F.cross_entropy
        pred_exp = pred.unsqueeze(2).expand(-1, -1, K, -1, -1) # (B, N, K, L, V)
        tgt_exp = tgt.unsqueeze(1).expand(-1, N, -1, -1)       # (B, N, K, L)
        
        pad_id = int(getattr(field, "pad_token_id", 0) or 0)
        
        loss = F.cross_entropy(
            pred_exp.reshape(-1, V),
            tgt_exp.reshape(-1),
            ignore_index=pad_id,
            reduction="none"
        )
        
        loss = loss.view(B, N, K, L)
        
        # Mask check
        mask = (tgt_exp != pad_id).float()
        denom = mask.sum(dim=-1).clamp_min(1.0)
        return (loss * mask).sum(dim=-1) / denom

    if isinstance(field, NumericDigitCategoryField):
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
        diff = pred.unsqueeze(2) - tgt.unsqueeze(1) # (B, N, K, 1)
        return diff.pow(2).mean(dim=-1)

    if isinstance(field, BooleanField):
        if getattr(field, "use_bce_loss", True):
            p_exp = pred.unsqueeze(2).expand(-1, -1, tgt.size(1), -1)
            t_exp = tgt.unsqueeze(1).expand(-1, pred.size(1), -1, -1)
            loss = F.binary_cross_entropy_with_logits(p_exp, t_exp, reduction="none")
            return loss.mean(dim=-1)
        else:
            diff = torch.tanh(pred.unsqueeze(2)) - tgt.unsqueeze(1)
            return diff.pow(2).mean(dim=-1)

    if isinstance(field, MultiCategoryField):
        p_exp = pred.unsqueeze(2).expand(-1, -1, tgt.size(1), -1)
        t_exp = tgt.unsqueeze(1).expand(-1, pred.size(1), -1, -1)
        loss = F.binary_cross_entropy_with_logits(p_exp, t_exp, reduction="none")
        return loss.mean(dim=-1)

    if isinstance(field, SingleCategoryField):
        B, N, C = pred.shape
        K = tgt.shape[1]
        p_exp = pred.unsqueeze(2).expand(-1, -1, K, -1)
        t_exp = tgt.unsqueeze(1).expand(-1, N, -1, -1).long().squeeze(-1)
        loss = F.cross_entropy(p_exp.reshape(-1, C), t_exp.reshape(-1), reduction="none")
        return loss.view(B, N, K)

    # Fallback
    diff = pred.unsqueeze(2) - tgt.unsqueeze(1)
    return diff.pow(2).mean(dim=-1)


def compute_cost_and_losses(
    people_ae,
    z_slots: torch.Tensor,
    Z_gt: torch.Tensor,
    Y_gt_fields: List[torch.Tensor],
    mask: torch.Tensor,
    w_latent: float,
    w_recon: float,
):
    """
    Computes the Dense Cost Matrices for matching AND the Loss tensors.
    
    Inputs:
        z_slots: (B, N, D) - Normalized Predicted Latents (Hypersphere)
        Z_gt:    (B, MaxK, D) - Normalized Ground Truth Latents (Hypersphere)
    
    Returns:
        C_total: (B, N, MaxK) for matching (detached)
        C_lat:   (B, N, MaxK) latent loss values (with grad)
        C_rec:   (B, N, MaxK) recon loss values (with grad)
        lengths: (B,) number of valid targets per batch
    """
    B, N, D = z_slots.shape
    
    # lengths of valid targets per sample
    # mask is (B, MaxK) bool
    lengths = mask.sum(dim=1).long()
    MaxK = Z_gt.shape[1]
    
    dec_device = next(people_ae.decoder.parameters()).device
    slot_device = z_slots.device

    # 1. Latent Cost (B, N, MaxK)
    # Sphere Awareness:
    # Since z_slots and Z_gt are normalized to length 1:
    # ||A - B||^2 = ||A||^2 + ||B||^2 - 2(A.B)
    #             = 1 + 1 - 2(CosineSimilarity)
    #             = 2 - 2(CosineSimilarity)
    # Therefore, minimizing Squared Euclidean Distance IS maximizing Cosine Similarity.
    # We use Squared Euclidean here as it's numerically friendly.
    
    diff_lat = z_slots.unsqueeze(2) - Z_gt.unsqueeze(1)
    C_lat_batch = diff_lat.pow(2).sum(dim=-1) 

    # 2. Recon Cost (B, N, MaxK)
    # Run decoder on all slots at once (B*N items)
    z_flat = z_slots.reshape(B * N, D).to(dec_device)
    dec_out = people_ae.decoder(z_flat) 

    C_rec_batch = torch.zeros((B, N, MaxK), device=dec_device)

    for fi, field in enumerate(people_ae.fields):
        pred_f = dec_out[fi]
        shape_suffix = pred_f.shape[1:]
        
        # Reshape to (B, N, ...)
        pred_f = pred_f.view(B, N, *shape_suffix)
        tgt_f = Y_gt_fields[fi].to(dec_device)
        
        # Vectorized loss matrix for this field
        loss_f = _field_loss_broadcast(field, pred_f, tgt_f)
        C_rec_batch += loss_f

    # Ensure everything is on the same device for combination
    C_lat_batch = C_lat_batch.to(dec_device)
    
    # 3. Weighted Combination for Matching
    # We detach because we don't backprop through the matching decision itself
    C_total_batch = (w_latent * C_lat_batch + w_recon * C_rec_batch).detach()
    
    # Return everything needed for matching + final loss gathering
    # We return C_lat_batch and C_rec_batch WITH gradients so we can gather later.
    return C_total_batch, C_lat_batch, C_rec_batch, lengths