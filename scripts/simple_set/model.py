# scripts/simple_set/model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.autoencoder.row_autoencoder import TransformerFieldDecoder


class PersonSetHead(nn.Module):
    def __init__(
        self,
        in_dim,
        person_dim,
        vocab_size,
        hidden_dim,
        local_to_global,
        dropout=0.1,
        init_scale=20.0,
    ):
        super().__init__()

        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, person_dim)

        self.use_res = in_dim == person_dim
        if self.use_res:
            self.res_scale = nn.Parameter(torch.tensor(0.1))

        self.bias = nn.Parameter(torch.empty(vocab_size))
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(init_scale))

        self.register_buffer("local_to_global", local_to_global.long(), persistent=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.constant_(self.in_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)
        nn.init.constant_(self.bias, -6.0)

    def forward(self, z, people_weight):
        x = self.in_proj(z)
        x = self.ln(x)
        x = self.act(x)
        x = self.drop(x)
        q = self.out_proj(x)

        if self.use_res:
            q = q + self.res_scale * z

        q = F.normalize(q, p=2, dim=-1)

        w = people_weight[self.local_to_global]
        w = F.normalize(w, p=2, dim=-1)

        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = F.linear(q, w) * scale + self.bias
        return logits


class HybridSetModel(nn.Module):
    def __init__(
        self,
        fields,
        num_people,
        heads_config,
        head_vocab_sizes,
        head_local_to_global,
        latent_dim=256,
        hidden_dim=1024,
        person_dim=None,
        dropout=0.1,
        num_movies=0,
        **kwargs,
    ):
        super().__init__()

        if num_movies <= 0:
            raise ValueError("num_movies must be > 0")

        self.fields = fields
        self.latent_dim = int(latent_dim)
        self.person_dim = int(person_dim) if person_dim is not None else self.latent_dim

        self.movie_embeddings = nn.Embedding(num_movies, self.latent_dim)
        nn.init.normal_(self.movie_embeddings.weight, std=0.02)

        self.person_embeddings = nn.Embedding(num_people, self.person_dim)
        nn.init.normal_(self.person_embeddings.weight, std=0.02)

        self.field_decoder = TransformerFieldDecoder(
            fields,
            self.latent_dim,
            num_layers=2,
            num_heads=4,
        )

        self.heads = nn.ModuleDict()
        init_scale = kwargs.get("hybrid_set_logit_scale", 20.0)

        for name, _ in heads_config.items():
            vocab = int(head_vocab_sizes.get(name, 0))
            if vocab <= 0:
                continue

            local_to_global = head_local_to_global.get(name)
            if local_to_global is None:
                continue

            self.heads[name] = PersonSetHead(
                in_dim=self.latent_dim,
                person_dim=self.person_dim,
                vocab_size=vocab,
                hidden_dim=hidden_dim,
                local_to_global=local_to_global,
                dropout=dropout,
                init_scale=init_scale,
            )

    def forward(self, field_tensors, batch_indices):
        if batch_indices is None:
            raise ValueError("batch_indices is required.")

        idx = batch_indices.to(self.movie_embeddings.weight.device, non_blocking=True)
        z = self.movie_embeddings(idx)
        z = F.normalize(z, p=2, dim=-1)

        recon_table = self.field_decoder(z)

        dummy_idx = idx.new_zeros(1)
        _ = self.person_embeddings(dummy_idx)

        people_weight = self.person_embeddings.weight

        logits_dict = {}
        for name, head_module in self.heads.items():
            logits_dict[name] = head_module(z, people_weight)

        return logits_dict, recon_table, z
