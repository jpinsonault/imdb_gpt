import torch
import torch.nn as nn
import torch.nn.functional as F


class SetDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_slots: int,
        hidden_mult: float = 2.0,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.num_slots = int(num_slots)
        h = max(self.latent_dim, int(self.latent_dim * hidden_mult))

        self.trunk = nn.Sequential(
            nn.Linear(self.latent_dim, h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.GELU(),
        )

        self.latent_head = nn.Linear(h, self.num_slots * self.latent_dim)
        self.presence_head = nn.Linear(h, self.num_slots)

    def forward(self, z_movie: torch.Tensor):
        h = self.trunk(z_movie)
        z_slots = self.latent_head(h)
        z_slots = z_slots.view(-1, self.num_slots, self.latent_dim)
        presence_logits = self.presence_head(h)
        return z_slots, presence_logits

    def predict(
        self,
        z_movie: torch.Tensor,
        threshold: float = 0.5,
        top_k: int | None = None,
    ):
        z_slots, presence_logits = self.forward(z_movie)
        probs = torch.sigmoid(presence_logits)

        if top_k is None:
            mask = probs > threshold
            if mask.sum(dim=-1).max().item() == 0:
                top_k = 1
            else:
                top_k = self.num_slots

        top_k = min(self.num_slots, int(top_k))
        probs_sorted, idx_sorted = probs.sort(dim=-1, descending=True)

        idx_top = idx_sorted[:, :top_k]
        probs_top = probs_sorted[:, :top_k]

        b = z_movie.size(0)
        z_out = []
        p_out = []
        for i in range(b):
            sel = probs_top[i] > threshold
            if sel.sum().item() == 0:
                sel = torch.zeros_like(probs_top[i], dtype=torch.bool)
                sel[0] = True
            chosen_idx = idx_top[i][sel]
            z_out.append(z_slots[i, chosen_idx])
            p_out.append(probs[i, chosen_idx])

        return z_out, p_out
