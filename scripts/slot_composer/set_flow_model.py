import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    def __init__(self, latent_dim: int, fourier_dim: int = 64):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(fourier_dim) * 2.0, requires_grad=False)
        self.proj_add = nn.Linear(2 * fourier_dim, latent_dim)
        self.proj_gamma = nn.Linear(2 * fourier_dim, latent_dim)
        self.proj_beta = nn.Linear(2 * fourier_dim, latent_dim)

    def forward(self, t: torch.Tensor):
        x = t.unsqueeze(-1) * self.freqs.unsqueeze(0)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        add = self.proj_add(x)
        gamma = self.proj_gamma(x)
        beta = self.proj_beta(x)
        return add, gamma, beta

class _TypedMHA(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.dim = int(dim)
        self.h = int(num_heads)
        self.dh = self.dim // self.h
        self.q_m = nn.Linear(dim, dim, bias=True)
        self.k_m = nn.Linear(dim, dim, bias=True)
        self.v_m = nn.Linear(dim, dim, bias=True)
        self.q_s = nn.Linear(dim, dim, bias=True)
        self.k_s = nn.Linear(dim, dim, bias=True)
        self.v_s = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.drop = nn.Dropout(dropout)

    def _split(self, x):
        b, t, d = x.shape
        x = x.view(b, t, self.h, self.dh).transpose(1, 2)
        return x

    def _merge(self, x):
        b, h, t, dh = x.shape
        x = x.transpose(1, 2).contiguous().view(b, t, h * dh)
        return x

    def forward(self, x):
        xm = x[:, :1, :]
        xs = x[:, 1:, :]
        q = torch.cat([self.q_m(xm), self.q_s(xs)], dim=1)
        k = torch.cat([self.k_m(xm), self.k_s(xs)], dim=1)
        v = torch.cat([self.v_m(xm), self.v_s(xs)], dim=1)
        q = self._split(q)
        k = self._split(k)
        v = self._split(v)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)
        out = attn @ v
        out = self._merge(out)
        out = self.proj(out)
        return out

class _TypedFlowBlock(nn.Module):
    def __init__(self, latent_dim: int, num_heads: int, ff_mult: float, dropout: float):
        super().__init__()
        self.attn = _TypedMHA(latent_dim, num_heads, dropout)
        self.ln1 = nn.LayerNorm(latent_dim)
        self.ffn_m = nn.Sequential(
            nn.Linear(latent_dim, int(ff_mult * latent_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(ff_mult * latent_dim), latent_dim),
            nn.Dropout(dropout),
        )
        self.ffn_s = nn.Sequential(
            nn.Linear(latent_dim, int(ff_mult * latent_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(ff_mult * latent_dim), latent_dim),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(latent_dim)

    def forward(self, tokens: torch.Tensor, add_t: torch.Tensor, gamma_t: torch.Tensor, beta_t: torch.Tensor):
        x = tokens + add_t
        a = self.attn(x)
        x = self.ln1(x + a)
        xm = x[:, :1, :]
        xs = x[:, 1:, :]
        fm = self.ffn_m(xm)
        fs = self.ffn_s(xs)
        f = torch.cat([fm, fs], dim=1)
        f = gamma_t * f + beta_t
        out = self.ln2(x + f)
        return out

class SetFlowSlotComposer(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_slots: int,
        num_heads: int,
        ff_mult: float,
        dropout: float,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.num_slots = int(num_slots)
        self.slots = nn.Parameter(torch.randn(num_slots, latent_dim) * 0.02)
        self.time = TimeEmbedding(latent_dim)
        self.block = _TypedFlowBlock(latent_dim, num_heads, ff_mult, dropout)
        self.out_norm = nn.LayerNorm(latent_dim)

    def forward(self, z_movie: torch.Tensor, steps: int, return_all: bool = False):
        b, d = z_movie.shape
        s0 = self.slots.unsqueeze(0).expand(b, -1, -1)
        tokens = torch.cat([z_movie.unsqueeze(1), s0], dim=1)
        seq = []
        for k in range(steps):
            t = 1.0 if steps <= 1 else float(k) / float(steps - 1)
            tt = tokens.new_full((b,), t)
            add_t, gamma_t, beta_t = self.time(tt)
            add_t = add_t.unsqueeze(1)
            gamma_t = gamma_t.unsqueeze(1)
            beta_t = beta_t.unsqueeze(1)
            tokens = self.block(tokens, add_t, gamma_t, beta_t)
            if return_all:
                seq.append(tokens)
        tokens = self.out_norm(tokens)
        z_m = tokens[:, 0, :]
        z_s = tokens[:, 1:, :]
        if return_all:
            inter = torch.stack(seq, dim=1)
            inter = self.out_norm(inter)
            inter_slots = inter[:, :, 1:, :]
            return z_m, z_s, inter_slots
        return z_m, z_s
