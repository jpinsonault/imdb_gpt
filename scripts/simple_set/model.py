# scripts/simple_set/model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.autoencoder.row_autoencoder import TransformerFieldDecoder


class SetHead(nn.Module):
    def __init__(
        self,
        in_dim,
        item_dim,
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
        self.out_proj = nn.Linear(hidden_dim, item_dim)

        self.use_res = in_dim == item_dim
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

    def forward(self, z, item_weight):
        x = self.in_proj(z)
        x = self.ln(x)
        x = self.act(x)
        x = self.drop(x)
        q = self.out_proj(x)

        if self.use_res:
            q = q + self.res_scale * z

        q = F.normalize(q, p=2, dim=-1)

        w = item_weight[self.local_to_global]
        w = F.normalize(w, p=2, dim=-1)

        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = F.linear(q, w) * scale + self.bias
        return logits


class HybridSetModel(nn.Module):
    def __init__(
        self,
        movie_fields,
        person_fields,
        num_movies,
        num_people,
        heads_config,
        movie_head_vocab_sizes,
        movie_head_local_to_global,
        person_head_vocab_sizes,
        person_head_local_to_global,
        movie_dim=256,
        hidden_dim=1024,
        person_dim=None,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__()

        if num_movies <= 0:
            raise ValueError("num_movies must be > 0")
        if num_people <= 0:
            raise ValueError("num_people must be > 0")

        self.movie_fields = movie_fields
        self.person_fields = person_fields

        self.movie_dim = int(movie_dim)
        self.person_dim = int(person_dim) if person_dim is not None else self.movie_dim

        self.movie_embeddings = nn.Embedding(num_movies, self.movie_dim)
        nn.init.normal_(self.movie_embeddings.weight, std=0.02)

        self.person_embeddings = nn.Embedding(num_people, self.person_dim)
        nn.init.normal_(self.person_embeddings.weight, std=0.02)

        self.movie_field_decoder = TransformerFieldDecoder(
            movie_fields,
            self.movie_dim,
            num_layers=2,
            num_heads=4,
        )

        self.person_field_decoder = TransformerFieldDecoder(
            person_fields,
            self.person_dim,
            num_layers=2,
            num_heads=4,
        )

        self.movie_heads = nn.ModuleDict()
        self.person_heads = nn.ModuleDict()

        init_scale = kwargs.get("hybrid_set_logit_scale", 20.0)

        for name, _ in heads_config.items():
            mvocab = int(movie_head_vocab_sizes.get(name, 0))
            m_l2g = movie_head_local_to_global.get(name)
            if mvocab > 0 and m_l2g is not None:
                self.movie_heads[name] = SetHead(
                    in_dim=self.movie_dim,
                    item_dim=self.person_dim,
                    vocab_size=mvocab,
                    hidden_dim=hidden_dim,
                    local_to_global=m_l2g,
                    dropout=dropout,
                    init_scale=init_scale,
                )

            pvocab = int(person_head_vocab_sizes.get(name, 0))
            p_l2g = person_head_local_to_global.get(name)
            if pvocab > 0 and p_l2g is not None:
                self.person_heads[name] = SetHead(
                    in_dim=self.person_dim,
                    item_dim=self.movie_dim,
                    vocab_size=pvocab,
                    hidden_dim=hidden_dim,
                    local_to_global=p_l2g,
                    dropout=dropout,
                    init_scale=init_scale,
                )

    def forward(self, movie_indices=None, person_indices=None):
        outputs = {}

        if movie_indices is not None:
            idx = movie_indices.to(self.movie_embeddings.weight.device, non_blocking=True).long()
            z = self.movie_embeddings(idx)
            z = F.normalize(z, p=2, dim=-1)

            recon_table = self.movie_field_decoder(z)

            _ = self.person_embeddings.weight
            people_weight = self.person_embeddings.weight

            logits_dict = {}
            for name, head_module in self.movie_heads.items():
                logits_dict[name] = head_module(z, people_weight)

            outputs["movie"] = (logits_dict, recon_table, z, idx)

        if person_indices is not None:
            idx_p = person_indices.to(self.person_embeddings.weight.device, non_blocking=True).long()
            z_p = self.person_embeddings(idx_p)
            z_p = F.normalize(z_p, p=2, dim=-1)

            recon_person = self.person_field_decoder(z_p)

            _ = self.movie_embeddings.weight
            movie_weight = self.movie_embeddings.weight

            logits_person = {}
            for name, head_module in self.person_heads.items():
                logits_person[name] = head_module(z_p, movie_weight)

            outputs["person"] = (logits_person, recon_person, z_p, idx_p)

        return outputs
