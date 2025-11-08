# scripts/path_siren/model.py

import math
import torch
import torch.nn as nn
from .dyn_linear import DynLinearRowCol, _CondMLP


class Sine(nn.Module):
    def __init__(self, w0: float):
        super().__init__()
        self.w0 = float(w0)

    def forward(self, x):
        return torch.sin(self.w0 * x)


class CondSirenLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        cond_dim: int,
        w0: float,
        is_first: bool,
        hidden: int,
        scale: float = 0.1,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.cond_dim = int(cond_dim)
        self.scale = float(scale)

        self.linear = nn.Linear(self.in_dim, self.out_dim)
        self.beta_in = _CondMLP(self.cond_dim, self.in_dim, hidden)
        self.beta_out = _CondMLP(self.cond_dim, self.out_dim, hidden)
        self.act = Sine(w0)

        self._init_weights(is_first, self.in_dim, float(w0))

    def _init_weights(self, is_first: bool, in_dim: int, w0: float):
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_dim, 1.0 / in_dim)
            else:
                b = math.sqrt(6.0 / in_dim) / max(w0, 1e-8)
                self.linear.weight.uniform_(-b, b)
            self.linear.bias.zero_()

    def forward(self, x, cond):
        b, l, _ = x.shape

        gamma_in = torch.tanh(self.beta_in(cond)).unsqueeze(1)
        x_mod = x * (1.0 + self.scale * gamma_in)

        y = self.linear(x_mod)

        gamma_out = torch.tanh(self.beta_out(cond)).unsqueeze(1)
        y = y * (1.0 + self.scale * gamma_out)

        return self.act(y)


class PathSiren(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_mult: float,
        layers: int,
        w0_first: float,
        w0_hidden: float,
        time_fourier: int = 0,
    ):
        super().__init__()

        d = int(latent_dim)
        l = int(max(2, layers))
        hm = float(hidden_mult)

        trunk = max(1, int(d * hm))
        cond = max(1, int(d * hm))

        self.d = d
        self.c = cond
        self.h = trunk
        self.layers = l
        self.t_fourier = max(0, int(time_fourier))

        self.proj_in = DynLinearRowCol(d, self.c, cond_dim=d, hidden=self.h)

        t_feats = 1 + 2 * self.t_fourier
        siren_in = t_feats + self.c

        self.first = CondSirenLayer(
            in_dim=siren_in,
            out_dim=self.h,
            cond_dim=self.c,
            w0=float(w0_first),
            is_first=True,
            hidden=self.h,
        )

        self.hiddens = nn.ModuleList(
            [
                CondSirenLayer(
                    in_dim=self.h,
                    out_dim=self.h,
                    cond_dim=self.c,
                    w0=float(w0_hidden),
                    is_first=False,
                    hidden=self.h,
                )
                for _ in range(l - 2)
            ]
        )

        self.last = nn.Linear(self.h, d)
        self.proj_out = DynLinearRowCol(d, d, cond_dim=d, hidden=self.h)

        with torch.no_grad():
            self.last.weight.zero_()
            self.last.bias.zero_()

    def _time_features(self, t01: torch.Tensor) -> torch.Tensor:
        if self.t_fourier <= 0:
            return 2.0 * t01 - 1.0
        parts = [2.0 * t01 - 1.0]
        two_pi = 2.0 * math.pi
        for i in range(self.t_fourier):
            f = float(2 ** i)
            ang = two_pi * f * t01
            parts.append(torch.sin(ang))
            parts.append(torch.cos(ang))
        return torch.cat(parts, dim=1)

    def forward(self, z_title: torch.Tensor, t_grid: torch.Tensor):
        b, d = z_title.shape
        l = int(t_grid.size(1))

        z_full = z_title
        z_proj = self.proj_in(z_full, z_full)

        t01 = t_grid.view(b * l, 1)
        t_feat = self._time_features(t01).view(b, l, -1)

        zc = z_proj.unsqueeze(1).expand(b, l, self.c)
        x0 = torch.cat([t_feat, zc], dim=2)

        h = self.first(x0, z_proj)
        for layer in self.hiddens:
            h = layer(h, z_proj)

        y_last = self.last(h)
        y_flat = y_last.contiguous().view(b * l, d)

        cond_out = z_full.repeat_interleave(l, dim=0)
        y_flat = self.proj_out(y_flat, cond_out)

        y = y_flat.view(b, l, d)
        base = z_title.unsqueeze(1).expand(b, l, d)
        z = base + y
        return z
