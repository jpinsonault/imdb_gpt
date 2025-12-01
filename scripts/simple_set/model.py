# scripts/simple_set/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scripts.autoencoder.row_autoencoder import TransformerFieldDecoder


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
        )
        nn.init.constant_(self.net[-2].weight, 0)

    def forward(self, x):
        return x + self.net(x)


class HybridSetModel(nn.Module):
    def __init__(
        self,
        fields: list,
        num_people: int,
        heads_config: dict,
        head_vocab_sizes: dict,
        latent_dim: int = 128,
        hidden_dim: int = 1024,
        base_output_rank: int = 64,
        depth: int = 12,
        dropout: float = 0.0,
        num_movies: int = 0,
    ):
        super().__init__()
        self.fields = fields
        self.heads_config = heads_config
        self.latent_dim = latent_dim

        if num_movies <= 0:
            raise ValueError("num_movies must be > 0 for Embedding Table mode.")

        self.movie_embeddings = nn.Embedding(num_movies, latent_dim)
        nn.init.normal_(self.movie_embeddings.weight, std=0.02)

        self.field_decoder = TransformerFieldDecoder(fields, latent_dim, num_layers=2, num_heads=4)

        self.trunk_proj = nn.Linear(latent_dim, hidden_dim)
        self.trunk = nn.Sequential(*[ResBlock(hidden_dim, dropout) for _ in range(depth)])

        self.people_bottlenecks = nn.ModuleDict()
        self.people_expansions = nn.ModuleDict()
        self.count_heads = nn.ModuleDict()

        for name, rank_mult in heads_config.items():
            rank = max(8, int(base_output_rank * rank_mult))
            vocab = head_vocab_sizes.get(name, num_people)

            self.people_bottlenecks[name] = nn.Linear(hidden_dim, rank, bias=False)
            self.people_expansions[name] = nn.Linear(rank, vocab)

            self.count_heads[name] = nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.GELU(),
                nn.Linear(256, 1),
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            if m is not getattr(self, "movie_embeddings", None):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

        for name, layer in self.people_expansions.items():
            if m is layer:
                prior_prob = 0.01
                bias_value = -math.log((1 - prior_prob) / prior_prob)
                nn.init.constant_(m.bias, bias_value)

    def forward(
        self,
        field_tensors: list,
        batch_indices: torch.Tensor,
        return_embeddings: bool = False,
    ):
        if batch_indices is None:
            raise ValueError("batch_indices is required for Embedding Table lookup.")

        idx = batch_indices.to(self.movie_embeddings.weight.device, non_blocking=True)
        raw_z_table = self.movie_embeddings(idx)
        z_table = F.normalize(raw_z_table, p=2, dim=-1)

        recon_table = self.field_decoder(z_table)

        feat = self.trunk_proj(z_table)
        feat = self.trunk(feat)

        logits_dict = {}
        counts_dict = {}

        for name in self.people_bottlenecks.keys():
            bn = self.people_bottlenecks[name](feat)
            if return_embeddings:
                logits_dict[name] = bn
            else:
                logits = self.people_expansions[name](bn)
                logits_dict[name] = logits

            cnt = self.count_heads[name](feat)
            counts_dict[name] = cnt

        return logits_dict, counts_dict, recon_table, z_table
