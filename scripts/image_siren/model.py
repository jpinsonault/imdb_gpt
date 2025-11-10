import math
import torch
import torch.nn as nn


class Sine(nn.Module):
    def __init__(self, w0: float):
        super().__init__()
        self.w0 = float(w0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)

        with torch.no_grad():
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.zeros_(self.fc1.bias)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(x)))


class StyleEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        style_dim: int,
        hidden_mult: float = 2.0,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.style_dim = int(style_dim)

        h = max(self.style_dim, int(self.latent_dim * float(hidden_mult)))
        if h <= 0:
            h = self.latent_dim

        self.in_proj = nn.Linear(self.latent_dim, h)
        self.blocks = nn.ModuleList(
            ResBlock(h) for _ in range(max(0, int(num_blocks)))
        )
        self.out_proj = nn.Linear(h, self.style_dim)

        with torch.no_grad():
            nn.init.xavier_uniform_(self.in_proj.weight)
            nn.init.zeros_(self.in_proj.bias)
            nn.init.xavier_uniform_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(z)
        for blk in self.blocks:
            x = blk(x)
        style = self.out_proj(x)
        return style


class CondSirenLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        style_dim: int,
        is_first: bool,
        w0_first: float,
        w0_hidden: float,
        film_scale: float = 0.1,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.style_dim = int(style_dim)
        self.film_scale = float(film_scale)

        self.linear = nn.Linear(self.in_dim, self.out_dim)
        self.to_gamma_beta = nn.Linear(self.style_dim, 2 * self.out_dim)

        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / self.in_dim, 1.0 / self.in_dim)
            else:
                b = math.sqrt(6.0 / self.in_dim) / max(float(w0_hidden), 1e-8)
                self.linear.weight.uniform_(-b, b)
            self.linear.bias.zero_()

            nn.init.xavier_uniform_(self.to_gamma_beta.weight)
            nn.init.zeros_(self.to_gamma_beta.bias)

        self.act = Sine(float(w0_first) if is_first else float(w0_hidden))

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape

        h = self.linear(x)

        gamma_beta = self.to_gamma_beta(style).view(b, 1, 2 * self.out_dim)
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        gamma = 1.0 + self.film_scale * gamma
        beta = self.film_scale * beta

        h = h * gamma + beta
        y = self.act(h)
        return y


class ImageCondSiren(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        out_channels: int = 3,
        hidden_dim: int = 256,
        hidden_layers: int = 20,
        w0_first: float = 30.0,
        w0_hidden: float = 1.0,
        style_dim: int | None = None,
        style_hidden_mult: float = 2.0,
        style_blocks: int = 2,
        film_scale: float = 0.1,
    ):
        super().__init__()

        self.latent_dim = int(latent_dim)
        self.out_channels = int(out_channels)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = max(1, int(hidden_layers))

        if style_dim is None:
            base = max(self.hidden_dim // 4, self.latent_dim // 2)
            style_dim = max(8, base)
        self.style_dim = int(style_dim)

        self.style_encoder = StyleEncoder(
            latent_dim=self.latent_dim,
            style_dim=self.style_dim,
            hidden_mult=style_hidden_mult,
            num_blocks=style_blocks,
        )

        in_dim = 2
        layers = []

        layers.append(
            CondSirenLayer(
                in_dim=in_dim,
                out_dim=self.hidden_dim,
                style_dim=self.style_dim,
                is_first=True,
                w0_first=w0_first,
                w0_hidden=w0_hidden,
                film_scale=film_scale,
            )
        )

        for _ in range(self.hidden_layers - 1):
            layers.append(
                CondSirenLayer(
                    in_dim=self.hidden_dim,
                    out_dim=self.hidden_dim,
                    style_dim=self.style_dim,
                    is_first=False,
                    w0_first=w0_first,
                    w0_hidden=w0_hidden,
                    film_scale=film_scale,
                )
            )

        self.layers = nn.ModuleList(layers)
        self.final = nn.Linear(self.hidden_dim, self.out_channels)

        with torch.no_grad():
            nn.init.uniform_(self.final.weight, -1e-3, 1e-3)
            nn.init.zeros_(self.final.bias)

    def _make_style(self, latents: torch.Tensor) -> torch.Tensor:
        return self.style_encoder(latents)

    def forward(self, coords: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        style = self._make_style(latents)
        x = coords
        for layer in self.layers:
            x = layer(x, style)
        out = self.final(x)
        return out
