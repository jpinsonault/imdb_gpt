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
