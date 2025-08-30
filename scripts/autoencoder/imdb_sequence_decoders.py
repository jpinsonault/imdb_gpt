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
        num_layers: int = 4,
        num_heads: int = 8,
        ff_mult: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.latent_dim = int(latent_dim)
        self.pos_enc = _SinusoidalPositionalEncoding(self.latent_dim, self.seq_len)
        self.learned_q = nn.Parameter(torch.zeros(self.seq_len, self.latent_dim))
        nn.init.xavier_uniform_(self.learned_q)

        layer = nn.TransformerDecoderLayer(
            d_model=self.latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_mult * self.latent_dim,
            dropout=dropout,
            batch_first=False,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.post = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, ff_mult * self.latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * self.latent_dim, self.latent_dim),
        )
        self.norm = nn.LayerNorm(self.latent_dim)

    def forward(self, z_m: torch.Tensor) -> torch.Tensor:
        memory = z_m.unsqueeze(0)
        b = z_m.size(0)
        tgt = self.learned_q.unsqueeze(1).expand(self.seq_len, b, self.latent_dim)
        tgt = tgt + self.pos_enc(self.seq_len, b)
        out = self.decoder(tgt, memory)
        out = out.transpose(0, 1)
        out = out + self.post(out)
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
        num_layers: int = 4,
        num_heads: int = 8,
        ff_mult: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.movie_encoder = movie_encoder
        self.people_decoder = people_decoder
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
