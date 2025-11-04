# scripts/slot_composer/vector_field_model.py
import torch
import torch.nn as nn

class TimeFourier(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(out_dim) * 2.0, requires_grad=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = t.unsqueeze(-1) * self.freqs.unsqueeze(0)
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

class SlotVectorFieldAttn(nn.Module):
    def __init__(self, latent_dim: int, hidden_mult: float, layers: int, fourier_dim: int, num_slots: int, heads: int = 4):
        super().__init__()
        d = int(latent_dim)
        h = int(max(1, int(hidden_mult * d)))
        n = int(num_slots)

        self.num_slots = n
        self.time = TimeFourier(fourier_dim)
        self.time_proj = nn.Linear(2 * fourier_dim, d)

        self.pos = nn.Parameter(torch.randn(n + 1, d) * 0.02)
        self.token_type = nn.Parameter(torch.randn(2, d) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=int(heads),
            dim_feedforward=h,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(layers))
        self.out = nn.Linear(d, d)

    def forward(self, s: torch.Tensor, t: torch.Tensor, z_movie: torch.Tensor) -> torch.Tensor:
        b, n, d = s.shape

        tm = self.time_proj(self.time(t))
        tm = tm.unsqueeze(1).expand(b, n + 1, d)

        movie_tok = z_movie.unsqueeze(1)
        slot_toks = s
        tokens = torch.cat([movie_tok, slot_toks], dim=1)

        pos = self.pos.unsqueeze(0).expand(b, n + 1, d)
        type_ids = torch.cat([
            self.token_type[0].unsqueeze(0).unsqueeze(0).expand(b, 1, d),
            self.token_type[1].unsqueeze(0).unsqueeze(0).expand(b, n, d),
        ], dim=1)

        x = tokens + pos + type_ids + tm
        h = self.encoder(x)
        slot_h = h[:, 1:, :]
        ds = self.out(slot_h)
        return ds

class RK4Solver(nn.Module):
    def __init__(self, steps: int, t0: float, t1: float):
        super().__init__()
        self.steps = int(max(1, steps))
        self.t0 = float(t0)
        self.t1 = float(t1)

    def forward(self, f, s0: torch.Tensor, z_movie: torch.Tensor, return_all: bool = False):
        b, n, d = s0.shape
        s = s0
        traj = [] if return_all else None
        h = (self.t1 - self.t0) / float(self.steps)
        for k in range(self.steps):
            t = s0.new_full((b,), self.t0 + k * h)
            k1 = f(s, t, z_movie)
            k2 = f(s + 0.5 * h * k1, t + 0.5 * h, z_movie)
            k3 = f(s + 0.5 * h * k2, t + 0.5 * h, z_movie)
            k4 = f(s + h * k3, t + h, z_movie)
            s = s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if return_all:
                traj.append(s)
        if return_all:
            return s, torch.stack(traj, dim=1)
        return s, None

class SlotFlowODE(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_slots: int,
        hidden_mult: float,
        layers: int,
        fourier_dim: int,
        steps: int,
        t0: float,
        t1: float,
        noise_scale: float,
        seed_from_movie: bool = True,
        cond_width: float = 2.0,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.num_slots = int(num_slots)
        self.noise_scale = float(noise_scale)
        self.seed_from_movie = bool(seed_from_movie)
        self.field = SlotVectorFieldAttn(
            latent_dim=latent_dim,
            hidden_mult=hidden_mult,
            layers=layers,
            fourier_dim=fourier_dim,
            num_slots=num_slots,
            heads=4,
        )
        self.solver = RK4Solver(steps=steps, t0=t0, t1=t1)
        if self.seed_from_movie:
            self.seed_proj = nn.Linear(latent_dim, num_slots * latent_dim)

    def _seed(self, z_movie: torch.Tensor) -> torch.Tensor:
        b, d = z_movie.shape
        if self.seed_from_movie:
            base = self.seed_proj(z_movie).view(b, self.num_slots, self.latent_dim)
        else:
            base = torch.zeros(b, self.num_slots, self.latent_dim, device=z_movie.device)
        noise = torch.randn_like(base)
        nrm = noise.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        noise = noise / nrm
        return base + self.noise_scale * noise

    def forward(self, z_movie: torch.Tensor, return_all: bool = False):
        s0 = self._seed(z_movie)
        sT, seq = self.solver(self.field, s0, z_movie, return_all=return_all)
        return sT, seq, s0
