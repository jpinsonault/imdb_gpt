import math
import torch
import torch.nn as nn
from .dyn_linear import DynLinearRowCol


class Sine(nn.Module):
    def __init__(self, w0: float):
        super().__init__()
        self.w0 = float(w0)

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, w0: float, is_first: bool):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.act = Sine(w0)
        self._init(is_first, in_dim)

    def _init(self, is_first: bool, in_dim: int):
        with torch.no_grad():
            if is_first:
                self.lin.weight.uniform_(-1.0 / in_dim, 1.0 / in_dim)
            else:
                b = math.sqrt(6.0 / in_dim) / self.act.w0
                self.lin.weight.uniform_(-b, b)
            self.lin.bias.zero_()

    def forward(self, x):
        return self.act(self.lin(x))


class FiLM(nn.Module):
    def __init__(self, z_dim: int, hidden: int, num_layers: int, layer_widths: list[int]):
        super().__init__()
        self.num_layers = int(num_layers)
        self.layer_widths = list(layer_widths)
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        total = 0
        for w in self.layer_widths:
            total += 2 * w
        self.out = nn.Linear(hidden, total)
        with torch.no_grad():
            self.out.weight.zero_()
            self.out.bias.zero_()

    def forward(self, z):
        h = self.net(z)
        o = self.out(h)
        outs = []
        idx = 0
        for w in self.layer_widths:
            g = o[:, idx:idx + w]
            b = o[:, idx + w:idx + 2 * w]
            idx += 2 * w
            gamma = 1.0 + 0.1 * torch.tanh(g)
            beta = 0.1 * torch.tanh(b)
            outs.append((gamma, beta))
        return outs


def _mlp(in_dim: int, out_dim: int, hidden: int | None = None) -> nn.Sequential:
    h = hidden if hidden is not None else max(in_dim, out_dim)
    return nn.Sequential(
        nn.Linear(in_dim, h),
        nn.GELU(),
        nn.Linear(h, out_dim),
    )


class PathSiren(nn.Module):
    def __init__(self, latent_dim: int, hidden_mult: float, layers: int, w0_first: float, w0_hidden: float):
        super().__init__()
        d = int(latent_dim)
        L = int(max(2, layers))
        hm = float(hidden_mult)

        self.d = d
        base_c = max(1, d // 4)
        base_h = max(1, d // 2)
        self.c = max(1, int(base_c * hm))
        self.h = max(1, int(base_h * hm))
        self.layers = L

        self.proj_in = DynLinearRowCol(d, self.c, cond_dim=d, hidden=self.h)
        siren_in = 1 + self.c

        self.first = SirenLayer(siren_in, self.h, w0_first, True)
        self.hiddens = nn.ModuleList([SirenLayer(self.h, self.h, w0_hidden, False) for _ in range(L - 2)])
        self.last = nn.Linear(self.h, d)
        self.cond = FiLM(z_dim=self.c, hidden=self.h, num_layers=L - 1, layer_widths=[self.h] * (L - 1))
        self.proj_out = DynLinearRowCol(d, d, cond_dim=d, hidden=self.h)

        with torch.no_grad():
            self.last.weight.zero_()
            self.last.bias.zero_()

    def _apply_film(self, x, gamma, beta):
        return gamma.unsqueeze(1) * x + beta.unsqueeze(1)

    def forward(self, z_title: torch.Tensor, t_grid: torch.Tensor):
        B, D = z_title.shape
        L = t_grid.size(1)

        z_proj = self.proj_in(z_title, z_title)

        t = t_grid.reshape(B * L, 1)
        t = 2.0 * t - 1.0

        zc = z_proj.unsqueeze(1).expand(B, L, self.c).reshape(B * L, self.c)
        x = torch.cat([t, zc], dim=1)

        mods = self.cond(z_proj)

        h = self.first(x)
        g0, b0 = mods[0]
        h = self._apply_film(h.view(B, L, -1), g0, b0).reshape(B * L, -1)

        for i, lay in enumerate(self.hiddens):
            h = lay(h)
            gamma, beta = mods[i + 1]
            h = self._apply_film(h.view(B, L, -1), gamma, beta).reshape(B * L, -1)

        y = self.last(h).view(B, L, D)
        y = y.view(B * L, D)
        cond_flat = z_title.repeat_interleave(L, dim=0)
        y = self.proj_out(y, cond_flat)
        y = y.view(B, L, D)

        base = z_title.unsqueeze(1).expand(B, L, D)
        z = base + y
        return z
