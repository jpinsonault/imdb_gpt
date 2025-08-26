# scripts/autoencoder/one_to_many/model.py
from __future__ import annotations
import math
from typing import List
import torch
import torch.nn as nn

class _SinusoidalPositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_len: int,
    ):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe)

    def forward(
        self,
        length: int,
        batch_size: int,
    ) -> torch.Tensor:
        x = self.pe[:length]
        return x.unsqueeze(1).expand(length, batch_size, x.size(-1))

class _TransformerTrunk(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        seq_len: int,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pos = _SinusoidalPositionalEncoding(latent_dim, self.seq_len)
        layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_mult * latent_dim,
            dropout=dropout,
            batch_first=False,
        )
        self.dec = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, z_src: torch.Tensor) -> torch.Tensor:
        mem = z_src.unsqueeze(0)
        tgt = self.pos(self.seq_len, z_src.size(0))
        out = self.dec(tgt, mem)
        out = out.transpose(0, 1)
        return self.norm(out)

class OneToManyPredictor(nn.Module):
    def __init__(
        self,
        source_encoder: nn.Module,
        target_decoder: nn.Module,
        latent_dim: int,
        seq_len: int,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.source_encoder = source_encoder
        self.target_decoder = target_decoder
        self.latent_dim = int(latent_dim)
        self.seq_len = int(seq_len)
        self.trunk = _TransformerTrunk(
            latent_dim=self.latent_dim,
            seq_len=self.seq_len,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_mult=ff_mult,
            dropout=dropout,
        )

    def forward(self, source_inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        z = self.source_encoder(source_inputs)
        seq_z = self.trunk(z)
        b = seq_z.size(0)
        flat = seq_z.reshape(b * self.seq_len, self.latent_dim)
        outs = self.target_decoder(flat)
        seq_outs = []
        for y in outs:
            seq_outs.append(y.view(b, self.seq_len, *y.shape[1:]))
        return seq_outs
