import torch
import torch.nn as nn

class SlotComposer(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_slots: int,
        num_layers: int,
        num_heads: int,
        ff_mult: float,
        dropout: float,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.num_slots = int(num_slots)

        self.slots = nn.Parameter(torch.randn(num_slots, latent_dim) * 0.02)
        self.pos = nn.Parameter(torch.randn(num_slots + 1, latent_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=int(ff_mult * latent_dim),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, z_movie: torch.Tensor):
        b, d = z_movie.shape
        s = self.slots.unsqueeze(0).expand(b, -1, -1)
        tokens = torch.cat([z_movie.unsqueeze(1), s], dim=1)
        tokens = tokens + self.pos.unsqueeze(0)
        out = self.encoder(tokens)
        out = self.norm(out)
        z_movie_hat = out[:, 0, :]
        z_slots_hat = out[:, 1:, :]
        return z_movie_hat, z_slots_hat
