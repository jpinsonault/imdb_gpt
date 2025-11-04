from typing import List
import torch
from scripts.autoencoder.fields import (
    ScalarField,
    BooleanField,
    MultiCategoryField,
    SingleCategoryField,
    TextField,
    NumericDigitCategoryField,
)
import torch.nn.functional as F

def per_sample_field_loss(field, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    if pred.dim() == 3:
        if tgt.dim() == 3:
            tgt = tgt.argmax(dim=-1)
        if isinstance(field, TextField):
            b, l, v = pred.shape
            loss_flat = F.cross_entropy(
                pred.reshape(b * l, v),
                tgt.reshape(b * l),
                ignore_index=int(getattr(field, "pad_token_id")),
                reduction="none",
            )
            loss_bt = loss_flat.reshape(b, l)
            mask_tok = (tgt.reshape(b, l) != int(getattr(field, "pad_token_id"))).float()
            denom = mask_tok.sum(dim=1).clamp_min(1.0)
            return (loss_bt * mask_tok).sum(dim=1) / denom
        if isinstance(field, NumericDigitCategoryField):
            b, p, v = pred.shape
            loss_flat = F.cross_entropy(
                pred.reshape(b * p, v),
                tgt.reshape(b * p).long(),
                reduction="none",
            )
            return loss_flat.reshape(b, p).mean(dim=1)

    if isinstance(field, ScalarField):
        diff = pred - tgt
        return diff.pow(2).reshape(diff.size(0), -1).mean(dim=1) if diff.dim() > 1 else diff.pow(2)

    if isinstance(field, BooleanField):
        if getattr(field, "use_bce_loss"):
            loss = F.binary_cross_entropy_with_logits(pred, tgt, reduction="none")
            return loss.reshape(loss.size(0), -1).mean(dim=1) if loss.dim() > 1 else loss
        diff = torch.tanh(pred) - tgt
        return diff.pow(2).reshape(diff.size(0), -1).mean(dim=1) if diff.dim() > 1 else diff.pow(2)

    if isinstance(field, MultiCategoryField):
        loss = F.binary_cross_entropy_with_logits(pred, tgt, reduction="none")
        return loss.reshape(loss.size(0), -1).mean(dim=1) if loss.dim() > 1 else loss

    if isinstance(field, SingleCategoryField):
        b, c = pred.shape[0], pred.shape[-1]
        t = tgt.long().squeeze(-1)
        return F.cross_entropy(pred.reshape(b, c), t.reshape(b), reduction="none")

    diff = pred - tgt
    return diff.pow(2).reshape(diff.size(0), -1).mean(dim=1) if diff.dim() > 1 else diff.pow(2)

def aggregate_people_loss(
    fields,
    preds_per_field: List[torch.Tensor],
    tgts_per_field: List[torch.Tensor],
    slot_mask: torch.Tensor,
) -> torch.Tensor:
    b, n = slot_mask.shape
    total = torch.zeros(b, device=slot_mask.device)
    for f, p, t in zip(fields, preds_per_field, tgts_per_field):
        for i in range(n):
            ps = per_sample_field_loss(f, p[:, i, ...], t[:, i, ...])
            total = total + ps * slot_mask[:, i]
    denom = slot_mask.sum(dim=1).clamp_min(1.0)
    return (total / denom).mean()

def latent_alignment_loss(z_hat: torch.Tensor, z_true: torch.Tensor, slot_mask: torch.Tensor) -> torch.Tensor:
    z_hat_n = F.normalize(z_hat, dim=-1)
    z_true_n = F.normalize(z_true, dim=-1)
    cos = (z_hat_n * z_true_n).sum(dim=-1)
    d = 1.0 - cos
    b, n = slot_mask.shape
    mask = slot_mask
    if d.dim() == 1:
        d = d.view(b, n)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return ((d * mask).sum(dim=1) / denom).mean()

def diversity_penalty(z_slots: torch.Tensor) -> torch.Tensor:
    b, n, d = z_slots.shape
    if n <= 1:
        return z_slots.new_zeros(())
    z = F.normalize(z_slots, dim=-1)
    sim = torch.matmul(z, z.transpose(1, 2))
    eye = torch.eye(n, device=z.device).unsqueeze(0)
    offdiag = sim * (1.0 - eye)
    return offdiag.mean()
