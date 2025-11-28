# scripts/set_decoder/training.py

import torch
import torch.nn.functional as F
from typing import List, Tuple

from scripts.autoencoder.fields import (
    ScalarField,
    BooleanField,
    MultiCategoryField,
    SingleCategoryField,
    TextField,
    NumericDigitCategoryField,
)

def _field_loss_sequence(field, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """
    Computes loss sequence (B, T).
    pred: (B, T, ...)
    tgt: (B, T, ...)
    """
    if isinstance(field, TextField):
        B, T, L, V = pred.shape
        pad_id = int(getattr(field, "pad_token_id", 0) or 0)
        
        loss = F.cross_entropy(
            pred.reshape(-1, V),
            tgt.reshape(-1),
            ignore_index=pad_id,
            reduction="none"
        )
        loss = loss.view(B, T, L)
        mask = (tgt != pad_id).float()
        denom = mask.sum(dim=-1).clamp_min(1.0)
        return (loss * mask).sum(dim=-1) / denom

    if isinstance(field, NumericDigitCategoryField):
        B, T, P, V = pred.shape
        mask_id = int(field.mask_index) if field.mask_index is not None else -1000
        loss = F.cross_entropy(
            pred.reshape(-1, V),
            tgt.reshape(-1),
            ignore_index=mask_id,
            reduction="none"
        )
        loss = loss.view(B, T, P)
        return loss.mean(dim=-1)

    if isinstance(field, ScalarField):
        # pred (B, T, 1), tgt (B, T, 1)
        diff = pred - tgt
        return diff.pow(2).mean(dim=-1)

    if isinstance(field, BooleanField):
        if getattr(field, "use_bce_loss", True):
            loss = F.binary_cross_entropy_with_logits(pred, tgt, reduction="none")
            return loss.mean(dim=-1)
        else:
            diff = torch.tanh(pred) - tgt
            return diff.pow(2).mean(dim=-1)

    if isinstance(field, MultiCategoryField):
        loss = F.binary_cross_entropy_with_logits(pred, tgt, reduction="none")
        return loss.mean(dim=-1)

    if isinstance(field, SingleCategoryField):
        B, T, C = pred.shape
        loss = F.cross_entropy(pred.reshape(-1, C), tgt.long().reshape(-1), reduction="none")
        return loss.view(B, T)

    # Fallback
    diff = pred - tgt
    return diff.pow(2).mean(dim=-1)


def compute_sequence_losses(
    people_ae,
    z_pred: torch.Tensor,
    Z_gt: torch.Tensor,
    Y_gt_fields: List[torch.Tensor],
    mask: torch.Tensor,
    w_latent: float,
    w_recon: float,
):
    """
    Computes sequence-aligned losses.
    
    Inputs:
        z_pred: (B, T, D) - Predicted Latents
        Z_gt:   (B, T, D) - Ground Truth Latents
        mask:   (B, T) - Valid steps
    
    Returns:
        loss_lat: Scalar (mean over valid steps)
        loss_rec: Scalar (mean over valid steps)
    """
    B, T, D = z_pred.shape
    dec_device = next(people_ae.decoder.parameters()).device
    
    # 1. Latent Loss (Cosine distance equivalent)
    # ||A - B||^2 = 2 - 2cos(sim)
    diff_lat = z_pred - Z_gt
    # (B, T)
    lat_loss_seq = diff_lat.pow(2).sum(dim=-1)
    
    # 2. Recon Loss
    # Run decoder on batch
    z_flat = z_pred.reshape(B * T, D).to(dec_device)
    dec_out = people_ae.decoder(z_flat)

    rec_loss_seq = torch.zeros((B, T), device=dec_device)

    for fi, field in enumerate(people_ae.fields):
        pred_f = dec_out[fi]
        shape_suffix = pred_f.shape[1:]
        
        # Reshape to (B, T, ...)
        pred_f = pred_f.view(B, T, *shape_suffix)
        tgt_f = Y_gt_fields[fi].to(dec_device)
        
        loss_f = _field_loss_sequence(field, pred_f, tgt_f)
        rec_loss_seq += loss_f.to(dec_device)

    # Apply Mask
    # Move mask to dec_device
    mask_dev = mask.to(dec_device)
    lat_loss_seq = lat_loss_seq.to(dec_device)
    
    # Sum and divide by valid elements
    denom = mask_dev.sum().clamp_min(1.0)
    
    loss_lat_final = (lat_loss_seq * mask_dev).sum() / denom
    loss_rec_final = (rec_loss_seq * mask_dev).sum() / denom
    
    return loss_lat_final, loss_rec_final