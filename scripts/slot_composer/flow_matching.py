# scripts/slot_composer/flow_matching.py

import torch
from typing import Tuple
from .set_losses import _cosine_cost, _greedy_assign_indices, _gather_true_by_assign

def _sample_t(b: int, device) -> torch.Tensor:
    return torch.rand(b, device=device)

def _blend(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if a.dim() == 3:
        return a * (1.0 - t.view(-1, 1, 1)) + b * t.view(-1, 1, 1)
    return a * (1.0 - t.view(-1, 1)) + b * t.view(-1, 1)

def rectified_flow_loss(
    vector_field,
    z_movie: torch.Tensor,
    s0: torch.Tensor,
    z_tgt: torch.Tensor,
    t: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    b = s0.size(0)
    device = s0.device
    if t is None:
        t = _sample_t(b, device)
    s_t = _blend(s0, z_tgt, t)
    v_star = z_tgt - s0
    v_pred = vector_field(s_t, t, z_movie)
    diff = v_pred - v_star
    if diff.dim() == 3:
        if mask is not None:
            num = (diff.pow(2).sum(dim=-1) * mask).sum(dim=1)
            den = mask.sum(dim=1).clamp_min(1.0)
            return (num / den).mean()
        return diff.pow(2).mean()
    return diff.pow(2).mean()

def rectified_flow_loss_multi(
    vector_field,
    z_movie: torch.Tensor,
    s0: torch.Tensor,
    z_tgt: torch.Tensor,
    mask: torch.Tensor | None = None,
    t_samples: int = 1,
) -> torch.Tensor:
    if t_samples <= 1:
        return rectified_flow_loss(vector_field, z_movie, s0, z_tgt, t=None, mask=mask)
    b, n, d = s0.shape
    device = s0.device
    t = torch.rand(t_samples, b, device=device)
    s0_r = s0.unsqueeze(0).expand(t_samples, b, n, d)
    zt_r = z_tgt.unsqueeze(0).expand(t_samples, b, n, d)
    s_t = s0_r * (1.0 - t.view(t_samples, b, 1, 1)) + zt_r * t.view(t_samples, b, 1, 1)
    v_star = zt_r - s0_r
    zm_r = z_movie.unsqueeze(0).expand(t_samples, b, d)
    t_flat = t.reshape(t_samples * b)
    s_t_flat = s_t.reshape(t_samples * b, n, d)
    zm_flat = zm_r.reshape(t_samples * b, d)
    v_pred = vector_field(s_t_flat, t_flat, zm_flat)
    diff = v_pred - v_star.reshape_as(v_pred)
    if mask is not None:
        mask_r = mask.unsqueeze(0).expand(t_samples, b, n)
        num = (diff.pow(2).sum(dim=-1).reshape(t_samples, b, n) * mask_r).sum(dim=2)
        den = mask_r.sum(dim=2).clamp_min(1.0)
        per_t = (num / den).mean(dim=1)
        return per_t.mean()
    per_t = diff.pow(2).reshape(t_samples, b, n, d).mean(dim=(2, 3))
    return per_t.mean()

def rectified_flow_loss_matched(
    vector_field,
    z_movie: torch.Tensor,
    s0: torch.Tensor,
    z_true: torch.Tensor,
    mask: torch.Tensor | None = None,
    t: torch.Tensor | None = None,
) -> torch.Tensor:
    b, n, d = s0.shape
    k = z_true.size(1)
    cost = _cosine_cost(s0, z_true)
    assign = _greedy_assign_indices(cost, int(k))
    z_tgt = _gather_true_by_assign(z_true, assign)
    return rectified_flow_loss(vector_field, z_movie, s0, z_tgt, t=t, mask=mask)

def rectified_flow_loss_matched_multi(
    vector_field,
    z_movie: torch.Tensor,
    s0: torch.Tensor,
    z_true: torch.Tensor,
    mask: torch.Tensor | None = None,
    t_samples: int = 1,
) -> torch.Tensor:
    b, n, d = s0.shape
    k = z_true.size(1)
    cost = _cosine_cost(s0, z_true)
    assign = _greedy_assign_indices(cost, int(k))
    z_tgt = _gather_true_by_assign(z_true, assign)
    return rectified_flow_loss_multi(vector_field, z_movie, s0, z_tgt, mask=mask, t_samples=t_samples)

def euler_sample(
    vector_field,
    z_movie: torch.Tensor,
    s0: torch.Tensor,
    steps: int = 8,
    t0: float = 0.0,
    t1: float = 1.0,
) -> torch.Tensor:
    h = (t1 - t0) / float(max(1, steps))
    s = s0
    for k in range(steps):
        t = s0.new_full((s0.size(0),), t0 + k * h)
        ds = vector_field(s, t, z_movie)
        s = s + h * ds
    return s

def rectified_seed(
    z_movie: torch.Tensor,
    num_slots: int,
    latent_dim: int,
    noise_scale: float,
    seed_proj: torch.nn.Linear | None = None,
) -> torch.Tensor:
    b, d = z_movie.shape
    if seed_proj is not None:
        base = seed_proj(z_movie).view(b, num_slots, latent_dim)
    else:
        base = torch.zeros(b, num_slots, latent_dim, device=z_movie.device)
    noise = torch.randn_like(base)
    nrm = noise.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    noise = noise / nrm
    return base + noise_scale * noise
