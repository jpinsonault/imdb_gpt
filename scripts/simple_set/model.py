# scripts/simple_set/model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.autoencoder.row_autoencoder import TransformerFieldDecoder
from scripts.autoencoder.fields import TextField


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
        noise_std=0.0,
        decoder_num_layers=2,
        decoder_num_heads=4,
        decoder_ff_multiplier=4,
        decoder_dropout=0.1,
        decoder_norm_first=False,
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
        self.noise_std = float(noise_std)

        self.movie_embeddings = nn.Embedding(num_movies, self.movie_dim)
        nn.init.normal_(self.movie_embeddings.weight, std=0.02)

        self.person_embeddings = nn.Embedding(num_people, self.person_dim)
        nn.init.normal_(self.person_embeddings.weight, std=0.02)

        self.movie_field_decoder = TransformerFieldDecoder(
            movie_fields,
            self.movie_dim,
            num_layers=decoder_num_layers,
            num_heads=decoder_num_heads,
            ff_dim=self.movie_dim * decoder_ff_multiplier,
            dropout=decoder_dropout,
            norm_first=decoder_norm_first,
        )

        self.person_field_decoder = TransformerFieldDecoder(
            person_fields,
            self.person_dim,
            num_layers=decoder_num_layers,
            num_heads=decoder_num_heads,
            ff_dim=self.person_dim * decoder_ff_multiplier,
            dropout=decoder_dropout,
            norm_first=decoder_norm_first,
        )

        self.movie_heads = nn.ModuleDict()
        self.person_heads = nn.ModuleDict()

        for name, _ in heads_config.items():
            movie_vocab_size = int(movie_head_vocab_sizes.get(name, 0))
            movie_l2g_map = movie_head_local_to_global.get(name)
            if movie_vocab_size > 0 and movie_l2g_map is not None:
                self.movie_heads[name] = SetHead(
                    in_dim=self.movie_dim,
                    item_dim=self.person_dim,
                    vocab_size=movie_vocab_size,
                    hidden_dim=hidden_dim,
                    local_to_global=movie_l2g_map,
                    dropout=dropout,
                    init_scale=logit_scale,
                    film_bottleneck_dim=film_bottleneck_dim,
                )

            person_vocab_size = int(person_head_vocab_sizes.get(name, 0))
            person_l2g_map = person_head_local_to_global.get(name)
            if person_vocab_size > 0 and person_l2g_map is not None:
                self.person_heads[name] = SetHead(
                    in_dim=self.person_dim,
                    item_dim=self.movie_dim,
                    vocab_size=person_vocab_size,
                    hidden_dim=hidden_dim,
                    local_to_global=person_l2g_map,
                    dropout=dropout,
                    init_scale=logit_scale,
                    film_bottleneck_dim=film_bottleneck_dim,
                )

        # Search encoders: lightweight text → embedding space projections
        self.movie_title_encoder = None
        self.person_name_encoder = None

        for f in movie_fields:
            if isinstance(f, TextField) and f.name == "primaryTitle":
                self.movie_title_encoder = f.build_encoder(self.movie_dim)
                break

        for f in person_fields:
            if isinstance(f, TextField) and f.name == "primaryName":
                self.person_name_encoder = f.build_encoder(self.person_dim)
                break

    def forward(self, movie_indices=None, person_indices=None, film_scale=1.0):
        outputs = {}
        film_reg_total = None

        if movie_indices is not None:
            idx = movie_indices.to(self.movie_embeddings.weight.device, non_blocking=True).long()
            z = self.movie_embeddings(idx)
            z = F.normalize(z, p=2, dim=-1)

            if self.training and self.noise_std > 0:
                z = z + torch.randn_like(z) * self.noise_std
                z = F.normalize(z, p=2, dim=-1)

            recon_table = self.movie_field_decoder(z)

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

            if self.training and self.noise_std > 0:
                z_p = z_p + torch.randn_like(z_p) * self.noise_std
                z_p = F.normalize(z_p, p=2, dim=-1)

            recon_person = self.person_field_decoder(z_p)

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
                film_reg_total = film_reg_total + film_reg_p if film_reg_total is not None else film_reg_p

        if film_reg_total is not None:
            outputs["film_reg"] = film_reg_total

        return outputs

    def compute_search_encoder_loss(self, movie_indices, movie_title_tokens, person_indices, person_name_tokens):
        """Cosine loss between encoder output and (stop-gradient) embedding lookup."""
        loss = torch.tensor(0.0, device=self.movie_embeddings.weight.device)
        n = 0

        if self.movie_title_encoder is not None and movie_indices is not None and movie_title_tokens is not None:
            enc = self.movie_title_encoder(movie_title_tokens)
            enc = F.normalize(enc, p=2, dim=-1)
            with torch.no_grad():
                target = F.normalize(self.movie_embeddings(movie_indices), p=2, dim=-1)
            loss = loss + (1.0 - (enc * target).sum(dim=-1)).mean()
            n += 1

        if self.person_name_encoder is not None and person_indices is not None and person_name_tokens is not None:
            enc = self.person_name_encoder(person_name_tokens)
            enc = F.normalize(enc, p=2, dim=-1)
            with torch.no_grad():
                target = F.normalize(self.person_embeddings(person_indices), p=2, dim=-1)
            loss = loss + (1.0 - (enc * target).sum(dim=-1)).mean()
            n += 1

        if n > 0:
            loss = loss / n
        return loss

    @torch.no_grad()
    def encode_movie_query(self, title_tokens):
        """Encode tokenized title → L2-normalized vector in movie embedding space."""
        if self.movie_title_encoder is None:
            raise RuntimeError("No movie title encoder available")
        enc = self.movie_title_encoder(title_tokens)
        return F.normalize(enc, p=2, dim=-1)

    @torch.no_grad()
    def encode_person_query(self, name_tokens):
        """Encode tokenized name → L2-normalized vector in person embedding space."""
        if self.person_name_encoder is None:
            raise RuntimeError("No person name encoder available")
        enc = self.person_name_encoder(name_tokens)
        return F.normalize(enc, p=2, dim=-1)
