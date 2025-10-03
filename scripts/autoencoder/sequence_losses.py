# scripts/autoencoder/sequence_losses.py
import torch
import torch.nn.functional as F
from typing import List
from .fields import (
    BaseField,
    ScalarField,
    BooleanField,
    MultiCategoryField,
    SingleCategoryField,
    TextField,
    NumericDigitCategoryField,
)

def _per_sample_field_loss_seq(field: BaseField, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    if isinstance(field, ScalarField):
        diff = pred - tgt
        if diff.dim() > 2:
            diff = diff.flatten(2)
            return diff.pow(2).mean(dim=2)
        return diff.pow(2)

    if isinstance(field, BooleanField):
        if getattr(field, "use_bce_loss", True):
            loss = F.binary_cross_entropy_with_logits(pred, tgt, reduction="none")
            if loss.dim() > 2:
                loss = loss.flatten(2).mean(dim=2)
            return loss
        diff = torch.tanh(pred) - tgt
        if diff.dim() > 2:
            diff = diff.flatten(2).mean(dim=2)
        return diff.pow(2)

    if isinstance(field, MultiCategoryField):
        loss = F.binary_cross_entropy_with_logits(pred, tgt, reduction="none")
        if loss.dim() > 2:
            loss = loss.flatten(2).mean(dim=2)
        return loss

    if isinstance(field, SingleCategoryField):
        B, T = pred.shape[0], pred.shape[1]
        C = pred.shape[-1]
        t = tgt.long().squeeze(-1)
        loss = F.cross_entropy(pred.reshape(B * T, C), t.reshape(B * T), reduction="none")
        return loss.reshape(B, T)

    if isinstance(field, TextField):
        if pred.dim() != 4 or tgt.dim() != 3:
            raise RuntimeError(f"TextField expects pred (B,T,L,V) or (B,T,V,L) and tgt (B,T,L), got pred={tuple(pred.shape)} tgt={tuple(tgt.shape)}")
        vocab = field.tokenizer.get_vocab_size()
        if pred.shape[-1] == vocab and pred.shape[-2] == tgt.shape[-1]:
            logits = pred
        elif pred.shape[-2] == vocab and pred.shape[-1] == tgt.shape[-1]:
            logits = pred.transpose(-1, -2)
        else:
            raise RuntimeError(f"TextField logits/tgt mismatch: pred={tuple(pred.shape)} tgt={tuple(tgt.shape)} vocab={vocab}")
        B, T, L, V = logits.shape
        pad_id = int(field.pad_token_id)
        loss_flat = F.cross_entropy(
            logits.reshape(B * T * L, V),
            tgt.reshape(B * T * L),
            ignore_index=pad_id,
            reduction="none",
        ).reshape(B * T, L)
        mask_tok = (tgt.reshape(B * T, L) != pad_id).float()
        denom = mask_tok.sum(dim=1).clamp_min(1.0)
        loss_bt = (loss_flat * mask_tok).sum(dim=1) / denom
        return loss_bt.reshape(B, T)

    if isinstance(field, NumericDigitCategoryField):
        if pred.dim() != 4 or tgt.dim() != 3:
            raise RuntimeError(f"DigitCategory expects pred (B,T,P,V)/(B,T,V,P) and tgt (B,T,P), got pred={tuple(pred.shape)} tgt={tuple(tgt.shape)}")
        base = int(getattr(field, "base", 10))
        if pred.shape[-1] == base and pred.shape[-2] == tgt.shape[-1]:
            logits = pred
        elif pred.shape[-2] == base and pred.shape[-1] == tgt.shape[-1]:
            logits = pred.transpose(-1, -2)
        else:
            raise RuntimeError(f"DigitCategory logits/tgt mismatch: pred={tuple(pred.shape)} tgt={tuple(tgt.shape)} base={base}")
        B, T, P, V = logits.shape
        loss_flat = F.cross_entropy(
            logits.reshape(B * T * P, V),
            tgt.reshape(B * T * P).long(),
            reduction="none",
        ).reshape(B * T, P)
        return loss_flat.mean(dim=1).reshape(B, T)

    diff = pred - tgt
    if diff.dim() > 2:
        diff = diff.flatten(2).mean(dim=2)
    return diff.pow(2)

def _sequence_loss_and_breakdown(
    fields: List[BaseField],
    preds_seq,
    targets_seq,
    mask: torch.Tensor,
):
    total = 0.0
    field_losses = {}
    denom = mask.sum().clamp_min(1.0)
    for f, pred, tgt in zip(fields, preds_seq, targets_seq):
        per_bt = _per_sample_field_loss_seq(f, pred, tgt)
        wsum = (per_bt * mask).sum()
        val = wsum / denom
        total = total + val * float(f.weight)
        field_losses[f.name] = float(val.detach().cpu().item())
    return total, field_losses

def _info_nce_masked_rows(z_pred_seq: torch.Tensor, z_tgt_seq: torch.Tensor, mask: torch.Tensor, temperature: float) -> torch.Tensor:
    z_pred_seq = torch.nn.functional.normalize(z_pred_seq, dim=-1)
    z_tgt_seq = torch.nn.functional.normalize(z_tgt_seq, dim=-1)
    B, T, _ = z_pred_seq.shape
    losses = []
    for t in range(T):
        row_mask = mask[:, t] > 0.5
        k = int(row_mask.sum().item())
        if k < 2:
            continue
        zp = z_pred_seq[row_mask, t, :]
        zt = z_tgt_seq[row_mask, t, :]
        logits = (zp @ zt.t()) / temperature
        labels = torch.arange(k, device=logits.device)
        losses.append(F.cross_entropy(logits, labels))
    if losses:
        return torch.stack(losses).mean()
    return z_pred_seq.new_zeros(())
