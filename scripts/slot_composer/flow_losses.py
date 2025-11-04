# scripts/slot_composer/flow_losses.py
import torch
import torch.nn.functional as F

def straight_path_loss(z_seq: torch.Tensor, z_start: torch.Tensor, z_target: torch.Tensor, slot_mask: torch.Tensor):
    b, s, n, d = z_seq.shape
    mask = slot_mask.unsqueeze(1).unsqueeze(-1)
    mask = mask.expand(b, s, n, d)

    if s <= 1:
        return z_seq.new_zeros(())

    ts = torch.linspace(0.0, 1.0, steps=s, device=z_seq.device)
    ts = ts.view(1, s, 1, 1)
    lerp = z_start.unsqueeze(1) * (1.0 - ts) + z_target.unsqueeze(1) * ts

    diff = z_seq - lerp
    diff = diff.pow(2).sum(dim=-1)
    diff = diff * slot_mask.unsqueeze(1)
    denom = slot_mask.sum(dim=1).clamp_min(1.0).unsqueeze(1)
    return (diff.sum(dim=(1, 2)) / denom.squeeze(1)).mean()
