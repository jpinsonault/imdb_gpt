# scripts/autoencoder/imdb_sequence_decoders.py
import math
import torch
import torch.nn as nn

class _SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, length: int, batch_size: int) -> torch.Tensor:
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
        self.seq_len = seq_len
        self.pos_enc = _SinusoidalPositionalEncoding(latent_dim, seq_len)
        layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_mult * latent_dim,
            dropout=dropout,
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, z_m: torch.Tensor) -> torch.Tensor:
        memory = z_m.unsqueeze(0)
        tgt = self.pos_enc(self.seq_len, z_m.size(0))
        out = self.decoder(tgt, memory)
        out = out.transpose(0, 1)
        out = self.norm(out)
        return out


class MovieToPeopleSequencePredictor(nn.Module):
    def __init__(
        self,
        movie_encoder: nn.Module,
        people_decoder: nn.Module,
        latent_dim: int,
        seq_len: int,
        width: int | None = None,
        depth: int = 3,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.movie_encoder = movie_encoder
        self.people_decoder = people_decoder
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.trunk = _TransformerTrunk(
            latent_dim=latent_dim,
            seq_len=seq_len,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_mult=ff_mult,
            dropout=dropout,
        )

    def forward(self, movie_inputs):
        z_m = self.movie_encoder(movie_inputs)
        z_seq = self.trunk(z_m)
        b = z_seq.size(0)
        flat = z_seq.reshape(b * self.seq_len, self.latent_dim)
        outs = self.people_decoder(flat)
        seq_outs = []
        for y in outs:
            seq_outs.append(y.view(b, self.seq_len, *y.shape[1:]))
        return seq_outs
