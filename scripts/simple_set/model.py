import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.autoencoder.row_autoencoder import TransformerFieldDecoder
from scripts.autoencoder.fields import TextField


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
        fields,
        num_people,
        heads_config,
        head_vocab_sizes,
        head_group_offsets,
        num_groups,
        latent_dim=128,
        hidden_dim=1024,
        base_output_rank=64,
        depth=12,
        dropout=0.0,
        num_movies=0,
        title_noise_prob=0.05,
    ):
        super().__init__()

        self.fields = fields
        self.heads_config = heads_config
        self.latent_dim = latent_dim
        self.title_noise_prob = float(title_noise_prob)
        self.num_groups = int(max(1, num_groups))

        if num_movies <= 0:
            raise ValueError("num_movies must be > 0 for Embedding Table mode.")

        self.movie_embeddings = nn.Embedding(num_movies, latent_dim)
        nn.init.normal_(self.movie_embeddings.weight, std=0.02)

        self.field_decoder = TransformerFieldDecoder(
            fields,
            latent_dim,
            num_layers=2,
            num_heads=4,
        )

        self.trunk_proj = nn.Linear(latent_dim, hidden_dim)
        self.trunk = nn.Sequential(
            *[ResBlock(hidden_dim, dropout) for _ in range(depth)]
        )

        self.group_bottlenecks = nn.ModuleDict()
        self.group_expansions = nn.ModuleDict()
        self.count_heads = nn.ModuleDict()

        for name, rank_mult in heads_config.items():
            vocab = int(head_vocab_sizes.get(name, num_people))
            if vocab <= 0:
                raise ValueError(f"Head {name} has non-positive vocab size {vocab}")

            offsets_tensor = head_group_offsets.get(name)
            if offsets_tensor is None or int(offsets_tensor[-1]) != vocab:
                offsets_tensor = torch.tensor([0, vocab], dtype=torch.long)

            if offsets_tensor.ndim != 1:
                raise ValueError(f"head_group_offsets[{name}] must be 1D, got {offsets_tensor.shape}")

            if offsets_tensor[0].item() != 0:
                raise ValueError(f"head_group_offsets[{name}][0] must be 0")

            num_groups_for_head = int(offsets_tensor.numel() - 1)
            if num_groups_for_head <= 0:
                num_groups_for_head = 1

            base_rank = max(8, int(base_output_rank * rank_mult))
            rank_per_group = max(4, base_rank // num_groups_for_head)

            bottlenecks = nn.ModuleList()
            expansions = nn.ModuleList()

            for g in range(num_groups_for_head):
                bottlenecks.append(nn.Linear(hidden_dim, rank_per_group, bias=False))
                start = int(offsets_tensor[g].item())
                end = int(offsets_tensor[g + 1].item())
                size_g = max(1, end - start)
                expansions.append(nn.Linear(rank_per_group, size_g))

            self.group_bottlenecks[name] = bottlenecks
            self.group_expansions[name] = expansions

            self.count_heads[name] = nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.GELU(),
                nn.Linear(256, 1),
            )

        self.title_field_index = None
        self.title_encoder = None
        self.title_pad_id = None
        self.title_special_ids = []
        self.title_valid_ids = None

        title_field = None
        for idx, f in enumerate(self.fields):
            if isinstance(f, TextField) and f.name == "primaryTitle":
                self.title_field_index = idx
                title_field = f
                break

        if title_field is not None:
            self.title_encoder = title_field.build_encoder(latent_dim)
            tok = title_field.tokenizer
            if tok is not None:
                specials = list(getattr(tok, "special_tokens", []))
                special_ids = [tok.token_to_id(s) for s in specials]
                vocab_size = tok.get_vocab_size()
                valid_ids = [i for i in range(vocab_size) if i not in special_ids]
                if valid_ids:
                    self.title_valid_ids = torch.tensor(valid_ids, dtype=torch.long)
                self.title_pad_id = title_field.pad_token_id
                self.title_special_ids = special_ids

        self.apply(self._init_weights)

        for name, bottlenecks in self.group_bottlenecks.items():
            for layer in bottlenecks:
                if layer.weight is not None:
                    nn.init.xavier_uniform_(layer.weight)

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

        for _, expansions in getattr(self, "group_expansions", {}).items():
            for layer in expansions:
                if m is layer:
                    prior_prob = 0.01
                    bias_value = -math.log((1.0 - prior_prob) / prior_prob)
                    nn.init.constant_(m.bias, bias_value)

    def _apply_title_noise(self, tokens):
        if self.title_noise_prob <= 0.0:
            return tokens
        if self.title_valid_ids is None:
            return tokens

        device = tokens.device
        mask = torch.rand(tokens.shape, device=device) < self.title_noise_prob

        if self.title_pad_id is not None:
            mask = mask & (tokens != self.title_pad_id)
        for sid in self.title_special_ids:
            mask = mask & (tokens != sid)

        if not mask.any():
            return tokens

        flat_mask = mask.view(-1)
        num = int(flat_mask.sum().item())
        if num == 0:
            return tokens

        valid_ids = self.title_valid_ids.to(device)
        choice_idx = torch.randint(valid_ids.size(0), (num,), device=device)
        new_vals = valid_ids[choice_idx]

        out = tokens.clone()
        out_flat = out.view(-1)
        out_flat[flat_mask] = new_vals
        return out

    def encode_titles(self, field_tensors):
        if self.title_field_index is None or self.title_encoder is None:
            raise RuntimeError("Title encoder not configured")
        x = field_tensors[self.title_field_index]
        if self.training:
            x = self._apply_title_noise(x)
        return self.title_encoder(x)

    def forward(self, field_tensors, batch_indices, return_embeddings=False):
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

        for name, bottlenecks in self.group_bottlenecks.items():
            group_logits = []
            for g, bn in enumerate(bottlenecks):
                h_g = bn(feat)
                if return_embeddings:
                    group_logits.append(h_g)
                else:
                    exp = self.group_expansions[name][g]
                    group_logits.append(exp(h_g))

            logits_dict[name] = torch.cat(group_logits, dim=-1)

            cnt = self.count_heads[name](feat)
            counts_dict[name] = cnt

        return logits_dict, counts_dict, recon_table, z_table
