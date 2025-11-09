import math
import torch
import torch.nn as nn


class Sine(nn.Module):
    def __init__(self, w0: float):
        super().__init__()
        self.w0 = float(w0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class CondSirenLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        latent_dim: int,
        is_first: bool,
        w0_first: float,
        w0_hidden: float,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.latent_dim = int(latent_dim)

        self.linear = nn.Linear(self.in_dim, self.out_dim)
        self.film = nn.Linear(self.latent_dim, 2 * self.out_dim)

        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / self.in_dim, 1.0 / self.in_dim)
            else:
                b = math.sqrt(6.0 / self.in_dim) / max(float(w0_hidden), 1e-8)
                self.linear.weight.uniform_(-b, b)
            self.linear.bias.zero_()
            self.film.weight.zero_()
            self.film.bias.zero_()

        self.act = Sine(float(w0_first) if is_first else float(w0_hidden))

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape

        h = self.linear(x)

        gamma_beta = self.film(z).view(b, 1, 2 * self.out_dim)
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        h = h * (1.0 + gamma) + beta
        y = self.act(h)
        return y


class ImageCondSiren(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        out_channels: int = 3,
        hidden_dim: int = 256,
        hidden_layers: int = 5,
        w0_first: float = 30.0,
        w0_hidden: float = 30.0,
    ):
        super().__init__()

        self.latent_dim = int(latent_dim)
        self.out_channels = int(out_channels)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = max(1, int(hidden_layers))

        in_dim = 2

        layers = []
        layers.append(
            CondSirenLayer(
                in_dim=in_dim,
                out_dim=self.hidden_dim,
                latent_dim=self.latent_dim,
                is_first=True,
                w0_first=w0_first,
                w0_hidden=w0_hidden,
            )
        )

        for _ in range(self.hidden_layers - 1):
            layers.append(
                CondSirenLayer(
                    in_dim=self.hidden_dim,
                    out_dim=self.hidden_dim,
                    latent_dim=self.latent_dim,
                    is_first=False,
                    w0_first=w0_first,
                    w0_hidden=w0_hidden,
                )
            )

        self.layers = nn.ModuleList(layers)
        self.final = nn.Linear(self.hidden_dim, self.out_channels)

        with torch.no_grad():
            nn.init.uniform_(
                self.final.weight,
                -1e-3,
                1e-3,
            )
            nn.init.zeros_(self.final.bias)

    def forward(self, coords: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        x = coords
        for layer in self.layers:
            x = layer(x, latents)
        out = self.final(x)
        return out
