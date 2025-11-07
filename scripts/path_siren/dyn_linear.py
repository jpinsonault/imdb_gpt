import torch
import torch.nn as nn


class _CondMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int):
        super().__init__()
        h = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.GELU(),
            nn.Linear(h, out_dim),
        )
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()

    def forward(self, z):
        return self.net(z)


class DynLinearRowCol(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, cond_dim: int, hidden: int):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.base = nn.Linear(in_dim, out_dim)
        self.make_alpha = _CondMLP(cond_dim, out_dim, hidden)
        self.make_beta = _CondMLP(cond_dim, in_dim, hidden)

    def forward(self, x, cond):
        beta = torch.tanh(self.make_beta(cond))
        alpha = torch.tanh(self.make_alpha(cond))
        x_scaled = x * (1.0 + 0.1 * beta)
        y = self.base(x_scaled)
        y = y * (1.0 + 0.1 * alpha)
        return y
