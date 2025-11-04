from typing import List, Tuple
import torch
import torch.nn.functional as F
from scripts.slot_composer.losses import per_sample_field_loss
from scripts.autoencoder.fields import (
    ScalarField,
    BooleanField,
    MultiCategoryField,
    SingleCategoryField,
    TextField,
    NumericDigitCategoryField,
)

def _pairwise_field_cost(field, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    if pred.dim() == 3 and tgt.dim() == 3:
        b, n, _ = pred.shape
        k = tgt.size(1)
        P = pred.unsqueeze(2).expand(b, n, k, *pred.shape[2:])
        T = tgt.unsqueeze(1).expand(b, n, k, *tgt.shape[2:])
        P = P.reshape(b * n * k, *P.shape[3:])
        T = T.reshape(b * n * k, *T.shape[3:])
        loss = per_sample_field_loss(field, P, T)
        return loss.view(b, n, k)

    if pred.dim() == 2 and tgt.dim() == 2:
        b, n = pred.shape[0], pred.shape[1]
        k = tgt.size(1)
        P = pred.unsqueeze(2).expand(b, n, k, *pred.shape[2:])
        T = tgt.unsqueeze(1).expand(b, n, k, *tgt.shape[2:])
        P = P.reshape(b * n * k, *P.shape[3:])
        T = T.reshape(b * n * k, *T.shape[3:])
        loss = per_sample_field_loss(field, P, T)
        return loss.view(b, n, k)

    if pred.dim() >= 3 and tgt.dim() >= 3:
        b, n = pred.shape[0], pred.shape[1]
        k = tgt.size(1)
        P = pred.unsqueeze(2).expand(b, n, k, *pred.shape[2:])
        T = tgt.unsqueeze(1).expand(b, n, k, *tgt.shape[2:])
        P = P.reshape(b * n * k, *P.shape[3:])
        T = T.reshape(b * n * k, *T.shape[3:])
        loss = per_sample_field_loss(field, P, T)
        return loss.view(b, n, k)

    b = pred.shape[0]
    n = pred.shape[1]
    k = tgt.size(1)
    P = pred.unsqueeze(2).expand(b, n, k, *pred.shape[2:])
    T = tgt.unsqueeze(1).expand(b, n, k, *tgt.shape[2:])
    P = P.reshape(b * n * k, *P.shape[3:])
    T = T.reshape(b * n * k, *T.shape[3:])
    loss = per_sample_field_loss(field, P, T)
    return loss.view(b, n, k)

def _greedy_assign(cost: torch.Tensor, valid_k: int) -> torch.Tensor:
    device = cost.device
    n, k = cost.shape
    if valid_k == 0:
        return cost.new_zeros(())
    c = cost[:, :valid_k].clone()
    inf = torch.tensor(float("inf"), device=device, dtype=c.dtype)
    total = c.new_zeros(())
    steps = min(n, valid_k)
    row_mask = torch.zeros(n, dtype=torch.bool, device=device)
    col_mask = torch.zeros(valid_k, dtype=torch.bool, device=device)
    for _ in range(steps):
        c_masked = c.masked_fill(row_mask.unsqueeze(1), inf).masked_fill(col_mask.unsqueeze(0), inf)
        i = torch.argmin(c_masked.view(-1))
        r = int(i.item() // valid_k)
        col = int(i.item() % valid_k)
        v = c[r, col]
        if torch.isinf(v):
            break
        total = total + v
        row_mask[r] = True
        col_mask[col] = True
    matched = col_mask.sum().clamp_min(1)
    return total / matched

def hungarian_people_loss(
    fields,
    preds_per_field: List[torch.Tensor],
    tgts_per_field: List[torch.Tensor],
    slot_mask: torch.Tensor,
) -> torch.Tensor:
    b, n = slot_mask.shape
    batch_costs = []
    for idx in range(b):
        k = int(slot_mask[idx].sum().item())
        if k == 0:
            batch_costs.append(slot_mask.new_zeros(()))
            continue
        c_sum = None
        for f, P, T in zip(fields, preds_per_field, tgts_per_field):
            Pi = P[idx].unsqueeze(0)
            Ti = T[idx, :k].unsqueeze(0)
            cur = _pairwise_field_cost(f, Pi, Ti)[0]
            c_sum = cur if c_sum is None else c_sum + cur
        batch_costs.append(_greedy_assign(c_sum, k))
    return torch.stack(batch_costs).mean()


import torch
import torch.nn.functional as F

def _cosine_cost(z_pred: torch.Tensor, z_true: torch.Tensor) -> torch.Tensor:
    b, n, d = z_pred.shape
    k = z_true.size(1)
    zp = F.normalize(z_pred, dim=-1)
    zt = F.normalize(z_true, dim=-1)
    return 1.0 - torch.matmul(zp, zt.transpose(1, 2))

def _greedy_assign_indices(cost: torch.Tensor, valid_k: int) -> torch.Tensor:
    b, n, k = cost.shape
    out = torch.full((b, n), -1, dtype=torch.long, device=cost.device)
    if valid_k == 0:
        return out
    inf = torch.tensor(float("inf"), device=cost.device, dtype=cost.dtype)
    for i in range(b):
        c = cost[i, :, :valid_k].clone()
        row_mask = torch.zeros(n, dtype=torch.bool, device=cost.device)
        col_mask = torch.zeros(valid_k, dtype=torch.bool, device=cost.device)
        steps = min(n, valid_k)
        for _ in range(steps):
            cm = c.masked_fill(row_mask.unsqueeze(1), inf).masked_fill(col_mask.unsqueeze(0), inf)
            idx = torch.argmin(cm.view(-1))
            r = int(idx.item() // valid_k)
            cidx = int(idx.item() % valid_k)
            v = c[r, cidx]
            if torch.isinf(v):
                break
            out[i, r] = cidx
            row_mask[r] = True
            col_mask[cidx] = True
    return out

def _gather_true_by_assign(z_true: torch.Tensor, assign_idx: torch.Tensor) -> torch.Tensor:
    b, n, d = assign_idx.shape[0], z_true.size(1), z_true.size(2)
    gather_idx = assign_idx.clamp_min(0)
    take = z_true.gather(1, gather_idx.unsqueeze(-1).expand(b, n, d))
    mask = (assign_idx >= 0).float().unsqueeze(-1)
    return take * mask

def latent_alignment_loss_matched(z_pred: torch.Tensor, z_true: torch.Tensor, slot_mask: torch.Tensor) -> torch.Tensor:
    b, n, d = z_pred.shape
    k = z_true.size(1)
    valid_k = slot_mask.sum(dim=1).clamp_max(k).to(torch.long)
    cost = _cosine_cost(z_pred, z_true)
    assign = _greedy_assign_indices(cost, int(k))
    z_true_perm = _gather_true_by_assign(z_true, assign)
    zp = F.normalize(z_pred, dim=-1)
    zt = F.normalize(z_true_perm, dim=-1)
    dcos = 1.0 - (zp * zt).sum(dim=-1)
    num = (dcos * slot_mask).sum(dim=1)
    den = slot_mask.sum(dim=1).clamp_min(1.0)
    return (num / den).mean()

def straight_path_loss_matched(z_seq: torch.Tensor, z_start: torch.Tensor, z_true: torch.Tensor, slot_mask: torch.Tensor) -> torch.Tensor:
    b, s, n, d = z_seq.shape
    k = z_true.size(1)
    cost = _cosine_cost(z_seq[:, -1, :, :], z_true)
    assign = _greedy_assign_indices(cost, int(k))
    z_tgt = _gather_true_by_assign(z_true, assign)
    if s <= 1:
        return z_seq.new_zeros(())
    ts = torch.linspace(0.0, 1.0, steps=s, device=z_seq.device).view(1, s, 1, 1)
    lerp = z_start.unsqueeze(1) * (1.0 - ts) + z_tgt.unsqueeze(1) * ts
    diff = (z_seq - lerp).pow(2).sum(dim=-1)
    diff = diff * slot_mask.unsqueeze(1)
    den = slot_mask.sum(dim=1).clamp_min(1.0).unsqueeze(1)
    return (diff.sum(dim=(1, 2)) / den.squeeze(1)).mean()
