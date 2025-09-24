# scripts/autoencoder/many_to_many/model.py
from __future__ import annotations
import math
from typing import List, Tuple
import torch
import torch.nn as nn

class _SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, length: int, batch_size: int) -> torch.Tensor:
        x = self.pe[:length]
        return x.unsqueeze(1).expand(length, batch_size, x.size(-1))


class _SeqGenerator(nn.Module):
    def __init__(self, latent_dim: int, seq_len: int, num_layers: int = 4, num_heads: int = 8, ff_mult: int = 8, dropout: float = 0.1):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.seq_len = int(seq_len)
        self.pos = _SinusoidalPositionalEncoding(self.latent_dim, self.seq_len)
        self.q = nn.Parameter(torch.zeros(self.seq_len, self.latent_dim))
        nn.init.xavier_uniform_(self.q)
        layer = nn.TransformerDecoderLayer(
            d_model=self.latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_mult * self.latent_dim,
            dropout=dropout,
            batch_first=False,
            norm_first=True,
        )
        self.dec = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.post = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, ff_mult * self.latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * self.latent_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
        )

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        b = memory.size(0)
        tgt = self.q.unsqueeze(1).expand(self.seq_len, b, self.latent_dim)
        tgt = tgt + self.pos(self.seq_len, b)
        mem = memory.unsqueeze(0)
        out = self.dec(tgt, mem).transpose(0, 1)
        out = self.post(out)
        return out


def _flatten_time_inputs(field_tensors: List[torch.Tensor]) -> Tuple[List[torch.Tensor], int]:
    b, t = field_tensors[0].shape[0], field_tensors[0].shape[1]
    flat = []
    for x in field_tensors:
        if x.dim() == 3:
            flat.append(x.view(b * t, x.shape[-1]))
        elif x.dim() == 2:
            flat.append(x.view(b * t))
        else:
            flat.append(x.view(b * t, *x.shape[2:]))
    return flat, t


def _encode_sequence(encoder: nn.Module, field_tensors: List[torch.Tensor], latent_dim: int) -> torch.Tensor:
    flat, t = _flatten_time_inputs(field_tensors)
    z = encoder(flat)
    if z.dim() > 2:
        z = z.flatten(1)
    b = field_tensors[0].shape[0]
    z = z.view(b, t, latent_dim)
    return z


class ManyToManyModel(nn.Module):
    def __init__(
        self,
        movie_encoder: nn.Module,
        people_encoder: nn.Module,
        movie_decoder: nn.Module,
        people_decoder: nn.Module,
        latent_dim: int,
        seq_len_titles: int | None = None,
        seq_len_people: int | None = None,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_mult: int = 8,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.movie_encoder = movie_encoder
        self.people_encoder = people_encoder
        self.movie_decoder = movie_decoder
        self.people_decoder = people_decoder
        self.latent_dim = int(latent_dim)
        self.seq_len_titles = None if seq_len_titles is None else int(seq_len_titles)
        self.seq_len_people = None if seq_len_people is None else int(seq_len_people)

        self.pool_movies = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
            nn.LayerNorm(self.latent_dim),
        )
        self.pool_people = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
            nn.LayerNorm(self.latent_dim),
        )

        t_len = max(1, self.seq_len_titles or 1)
        p_len = max(1, self.seq_len_people or 1)
        self.gen_people_from_movies = _SeqGenerator(self.latent_dim, p_len, num_layers, num_heads, ff_mult, dropout)
        self.gen_titles_from_people = _SeqGenerator(self.latent_dim, t_len, num_layers, num_heads, ff_mult, dropout)

    def forward(self, x_movies_fields: List[torch.Tensor], x_people_fields: List[torch.Tensor]):
        """
        Returns:
          preds_titles_seq, preds_people_seq, z_movie_to_people, z_people_to_movies
        """
        z_titles_seq = _encode_sequence(self.movie_encoder, x_movies_fields, self.latent_dim)
        z_people_seq = _encode_sequence(self.people_encoder, x_people_fields, self.latent_dim)

        z_movie_big = self.pool_movies(z_titles_seq.mean(dim=1))
        z_people_big = self.pool_people(z_people_seq.mean(dim=1))

        z_movie_to_people = self.gen_people_from_movies(z_movie_big)
        z_people_to_movies = self.gen_titles_from_people(z_people_big)

        b = z_movie_to_people.size(0)
        tp = z_movie_to_people.size(1)
        tt = z_people_to_movies.size(1)

        flat_m2p = z_movie_to_people.reshape(b * tp, self.latent_dim)
        flat_p2m = z_people_to_movies.reshape(b * tt, self.latent_dim)

        dec_people = self.people_decoder(flat_m2p)
        preds_people_seq = [y.view(b, tp, *y.shape[1:]) for y in dec_people]

        dec_titles = self.movie_decoder(flat_p2m)
        preds_titles_seq = [y.view(b, tt, *y.shape[1:]) for y in dec_titles]

        return preds_titles_seq, preds_people_seq, z_movie_to_people, z_people_to_movies
