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
        dropout,
        init_scale,
        film_bottleneck_dim,
    ):
        super().__init__()

        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)

        self.in_proj = nn.Linear(self.in_dim, self.hidden_dim)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(self.hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(self.hidden_dim, item_dim)

        self.use_res = self.in_dim == item_dim
        if self.use_res:
            self.res_scale = nn.Parameter(torch.tensor(0.1))

        bottleneck_dim = int(film_bottleneck_dim)
        if bottleneck_dim <= 0:
            raise ValueError("film_bottleneck_dim must be > 0")

        self.film_bottleneck = nn.Linear(self.in_dim, bottleneck_dim)
        self.film_act = nn.GELU()
        self.film_out = nn.Linear(bottleneck_dim, self.hidden_dim * 2)

        self.bias = nn.Parameter(torch.empty(vocab_size))
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(init_scale))

        self.register_buffer("local_to_global", local_to_global.long(), persistent=False)

        self.last_reg = None

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.constant_(self.in_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

        nn.init.xavier_uniform_(self.film_bottleneck.weight)
        nn.init.constant_(self.film_bottleneck.bias, 0.0)
        nn.init.xavier_uniform_(self.film_out.weight)
        nn.init.constant_(self.film_out.bias, 0.0)

        nn.init.constant_(self.bias, -6.0)

    def _apply_film(self, x, z, film_scale):
        if film_scale <= 0.0:
            self.last_reg = None
            return x

        h = self.film_bottleneck(z)
        h = self.film_act(h)
        gb = self.film_out(h)
        gb = gb.view(z.size(0), 2, self.hidden_dim)

        gamma = gb[:, 0, :]
        beta = gb[:, 1, :]

        if self.training:
            gamma_centered = gamma - gamma.mean(dim=0, keepdim=True)
            beta_centered = beta - beta.mean(dim=0, keepdim=True)
            self.last_reg = gamma_centered.pow(2).mean() + beta_centered.pow(2).mean()
        else:
            self.last_reg = None

        gamma = gamma * float(film_scale)
        beta = beta * float(film_scale)

        return x * (1.0 + gamma) + beta

    def forward(self, z, item_weight, film_scale=1.0):
        x = self.in_proj(z)
        x = self.ln(x)
        x = self.act(x)
        x = self._apply_film(x, z, film_scale)
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
        movie_dim,
        hidden_dim,
        person_dim,
        dropout,
        logit_scale,
        film_bottleneck_dim,
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
                    init_scale=logit_scale,
                    film_bottleneck_dim=film_bottleneck_dim,
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
                    init_scale=logit_scale,
                    film_bottleneck_dim=film_bottleneck_dim,
                )

    def forward(self, movie_indices=None, person_indices=None, film_scale=1.0):
        outputs = {}
        film_reg_total = None

        if movie_indices is not None:
            idx = movie_indices.to(self.movie_embeddings.weight.device, non_blocking=True).long()
            z = self.movie_embeddings(idx)
            z = F.normalize(z, p=2, dim=-1)

            recon_table = self.movie_field_decoder(z)

            _ = self.person_embeddings.weight
            people_weight = self.person_embeddings.weight

            logits_dict = {}
            film_reg = None
            for name, head_module in self.movie_heads.items():
                logits = head_module(z, people_weight, film_scale=film_scale)
                logits_dict[name] = logits
                if head_module.last_reg is not None:
                    if film_reg is None:
                        film_reg = head_module.last_reg
                    else:
                        film_reg = film_reg + head_module.last_reg

            outputs["movie"] = (logits_dict, recon_table, z, idx)

            if film_reg is not None:
                film_reg_total = film_reg if film_reg_total is None else film_reg_total + film_reg

        if person_indices is not None:
            idx_p = person_indices.to(self.person_embeddings.weight.device, non_blocking=True).long()
            z_p = self.person_embeddings(idx_p)
            z_p = F.normalize(z_p, p=2, dim=-1)

            recon_person = self.person_field_decoder(z_p)

            _ = self.movie_embeddings.weight
            movie_weight = self.movie_embeddings.weight

            logits_person = {}
            film_reg_p = None
            for name, head_module in self.person_heads.items():
                logits = head_module(z_p, movie_weight, film_scale=film_scale)
                logits_person[name] = logits
                if head_module.last_reg is not None:
                    if film_reg_p is None:
                        film_reg_p = head_module.last_reg
                    else:
                        film_reg_p = film_reg_p + head_module.last_reg

            outputs["person"] = (logits_person, recon_person, z_p, idx_p)

            if film_reg_p is not None:
                film_reg_total = film_reg_p if film_reg_total is not None else film_reg_p

        if film_reg_total is not None:
            outputs["film_reg"] = film_reg_total

        return outputs
