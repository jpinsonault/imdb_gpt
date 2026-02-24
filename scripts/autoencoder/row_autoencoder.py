# scripts/autoencoder/row_autoencoder.py

import logging
import json
from pathlib import Path
from typing import Any, List, Dict, Tuple

import numpy as np
from config import ProjectConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

from .fields import BaseField

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class TransformerFieldDecoder(nn.Module):
    def __init__(
        self,
        fields: List[BaseField],
        latent_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int | None = None,
        dropout: float = 0.1,
        norm_first: bool = False,
    ):
        super().__init__()
        self.fields = fields
        self.latent_dim = int(latent_dim)
        self.num_fields = len(fields)

        self.global_token = nn.Parameter(torch.randn(1, self.latent_dim) * 0.02)
        self.field_tokens = nn.Parameter(torch.randn(self.num_fields, self.latent_dim) * 0.02)

        d_ff = int(ff_dim) if ff_dim is not None else self.latent_dim * 4

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            norm_first=norm_first,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.final_norm = nn.LayerNorm(self.latent_dim) if norm_first else None

        self.heads = nn.ModuleList(
            [f.build_decoder(self.latent_dim) for f in self.fields]
        )

        self._field_decoders = None

    def _decode_all_fields(self, z: torch.Tensor) -> List[torch.Tensor]:
        if z.dim() == 1:
            z = z.unsqueeze(0)

        b = z.size(0)

        g = self.global_token.expand(b, -1).unsqueeze(1)
        f = self.field_tokens.unsqueeze(0).expand(b, -1, -1)

        x = torch.cat([g, f], dim=1)
        x = x + z.unsqueeze(1)

        h = self.transformer(x)
        if self.final_norm is not None:
            h = self.final_norm(h)
        field_h = h[:, 1:, :]

        outs: List[torch.Tensor] = []
        for i, head in enumerate(self.heads):
            token_i = field_h[:, i, :]
            y = head(token_i)
            outs.append(y)

        return outs

    def forward(self, z: torch.Tensor) -> List[torch.Tensor]:
        return self._decode_all_fields(z)

    @property
    def field_decoders(self):
        """Per-field decoder callables. Each takes a latent vector and returns that field's output."""
        if self._field_decoders is None:
            decoders = []

            for field_index in range(self.num_fields):
                def make_decoder(idx: int):
                    def decode_field(z: torch.Tensor, idx: int = idx):
                        if z.dim() == 1:
                            z_in = z.unsqueeze(0)
                            single = True
                        else:
                            z_in = z
                            single = False

                        batch_size = z_in.size(0)

                        g = self.global_token.expand(batch_size, -1).unsqueeze(1)
                        f = self.field_tokens.unsqueeze(0).expand(batch_size, -1, -1)

                        x = torch.cat([g, f], dim=1)
                        x = x + z_in.unsqueeze(1)

                        h = self.transformer(x)
                        if self.final_norm is not None:
                            h = self.final_norm(h)
                        token_i = h[:, 1 + idx, :]

                        y = self.heads[idx](token_i)
                        if single:
                            return y[0]
                        return y

                    return decode_field

                decoders.append(make_decoder(field_index))

            self._field_decoders = decoders

        return self._field_decoders

def _out_dim(m: nn.Module) -> int:
    for p in reversed(list(m.parameters())):
        if p.dim() == 2:
            return p.size(0)
    return 0


class _FieldEncoders(nn.Module):
    def __init__(self, fields: List[BaseField], latent_dim: int):
        super().__init__()
        self.fields = fields
        self.latent_dim = int(latent_dim)
        self.num_fields = len(fields)
        self.num_tokens = self.num_fields + 1

        self.encs = nn.ModuleList(
            [f.build_encoder(self.latent_dim) for f in self.fields]
        )
        self.proj = nn.ModuleList(
            [
                nn.Identity() if _out_dim(m) == self.latent_dim
                else nn.Linear(_out_dim(m), self.latent_dim)
                for m in self.encs
            ]
        )

        self.fuse = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=4,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(self.latent_dim)

        self.field_embed = nn.Parameter(
            torch.randn(self.num_fields, self.latent_dim) * 0.02
        )
        self.cls_token = nn.Parameter(
            torch.randn(1, self.latent_dim) * 0.02
        )

        self.q_bias = nn.Parameter(
            torch.zeros(self.num_tokens, self.latent_dim)
        )
        self.k_bias = nn.Parameter(
            torch.zeros(self.num_tokens, self.latent_dim)
        )

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        outs = []
        for x, enc, proj in zip(xs, self.encs, self.proj):
            y = enc(x)
            if y.dim() > 2:
                y = y.flatten(1)
            y = proj(y)
            outs.append(y)

        if not outs:
            raise RuntimeError("No fields provided to _FieldEncoders")

        field_tokens = torch.stack(outs, dim=1)
        field_tokens = field_tokens + self.field_embed.unsqueeze(0)

        b = field_tokens.size(0)
        cls = self.cls_token.unsqueeze(0).expand(b, 1, -1)

        all_tokens = torch.cat([cls, field_tokens], dim=1)

        q = all_tokens + self.q_bias.unsqueeze(0)
        k = all_tokens + self.k_bias.unsqueeze(0)

        h, _ = self.fuse(q, k, all_tokens)

        cls_h = h[:, 0, :]
        z = self.norm(cls_h)
        return z


class _RowDataset(IterableDataset):
    def __init__(self, row_gen_fn, fields: List[BaseField]):
        super().__init__()
        self.row_gen_fn = row_gen_fn
        self.fields = fields

    def __iter__(self):
        for row in self.row_gen_fn():
            xs = [f.transform(row.get(f.name)) for f in self.fields]
            ys = [f.transform_target(row.get(f.name)) for f in self.fields]
            yield xs, ys


def _collate(
    batch: List[Tuple[List[torch.Tensor], List[torch.Tensor]]]
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    x_cols = list(zip(*[b[0] for b in batch]))
    y_cols = list(zip(*[b[1] for b in batch]))
    X = [torch.stack(col, dim=0) for col in x_cols]
    Y = [torch.stack(col, dim=0) for col in y_cols]
    return X, Y


def _field_to_state(f: BaseField) -> Dict[str, Any]:
    return f.get_state()


def _apply_field_state(f: BaseField, st: Dict[str, Any]) -> None:
    f.set_state(st)


class RowAutoencoder:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.model_dir = Path(config.model_dir)
        self.model = None
        self.encoder = None
        self.decoder = None

        self.fields: List[BaseField] = self.build_fields()
        self.latent_dim = self.config.latent_dim
        self.num_rows_in_dataset = 0

        self.db_path: str = config.db_path
        self.stats_accumulated = False

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    def _get_cache_path(self) -> Path:
        """Return path to the JSON cache file for this autoencoder class."""
        return Path(self.config.data_dir) / f"{self.__class__.__name__}_stats.json"

    def _drop_cache(self):
        """Delete the JSON stats cache file."""
        p = self._get_cache_path()
        if p.exists():
            try:
                p.unlink()
                logging.info(f"Deleted cache file {p}")
            except OSError as e:
                logging.warning(f"Failed to delete cache file {p}: {e}")

    def _save_cache(self):
        """Save field stats to a JSON file."""
        cache_data = {}
        for f in self.fields:
            cache_data[f.name] = _field_to_state(f)
        
        p = self._get_cache_path()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, 'w') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Failed to save stats cache to {p}: {e}")

    def _load_cache(self) -> bool:
        """Load field stats from a JSON file."""
        p = self._get_cache_path()
        if not p.exists():
            return False
        try:
            with open(p, 'r') as f:
                cache_data = json.load(f)
            
            for f in self.fields:
                if f.name in cache_data:
                    _apply_field_state(f, cache_data[f.name])
            
            self.stats_accumulated = True
            return True
        except Exception as e:
            logging.warning(f"Failed to load stats cache from {p}: {e}")
            return False

    def accumulate_stats(self):
        use_cache = self.config.use_cache
        refresh_cache = self.config.refresh_cache
        if refresh_cache:
            self._drop_cache()
        if use_cache and self._load_cache():
            logging.info("stats loaded from cache")
            return
        if self.stats_accumulated:
            logging.info("stats already accumulated")
            return
        n = 0
        logging.info("accumulating stats")
        for row in tqdm(
            self.row_generator(),
            desc=self.__class__.__name__,
        ):
            self.accumulate_stats_for_row(row)
            n += 1
        self.num_rows_in_dataset = n
        logging.info(f"stats accumulation finished ({n} rows)")

    def finalize_stats(self):
        if self.stats_accumulated:
            return
        logging.info("finalizing stats")
        for f in self.fields:
            f.finalize_stats()
        self.stats_accumulated = True
        self._save_cache()
        logging.info("stats finalized and cached")

    def accumulate_stats_for_row(self, row: Dict):
        for f in self.fields:
            f.accumulate_stats(row.get(f.name))

    def transform_row(self, row: Dict) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for f in self.fields:
            out[f.name] = f.transform(row.get(f.name))
        return out

    def reconstruct_row(self, latent_vector: np.ndarray) -> Dict[str, str]:
        if self.decoder is None:
            raise RuntimeError("Decoder not built")
        if not self.stats_accumulated:
            raise RuntimeError("Field stats must be finalized")
        z = torch.tensor(
            latent_vector,
            dtype=torch.float32,
            device=self.device,
        )
        if z.dim() == 1:
            z = z.unsqueeze(0)
        with torch.no_grad():
            outs = self.decoder(z)
        rec: Dict[str, str] = {}
        for f, o in zip(self.fields, outs):
            # Use render_prediction as this comes from the model decoder
            rec[f.name] = f.render_prediction(o[0])
        return rec

    def build_fields(self) -> List["BaseField"]:
        raise NotImplementedError

    def row_generator(self):
        raise NotImplementedError

    def build_autoencoder(self):
        if not self.stats_accumulated:
            raise RuntimeError(
                "Call accumulate_stats()/finalize_stats() first"
            )

        self.encoder = _FieldEncoders(
            self.fields,
            self.latent_dim,
        ).to(self.device)

        self.decoder = TransformerFieldDecoder(
            self.fields,
            self.latent_dim,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
            norm_first=False,
        ).to(self.device)

        class _AE(nn.Module):
            def __init__(self, enc, dec):
                super().__init__()
                self.enc = enc
                self.dec = dec

            def forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
                z = self.enc(xs)
                outs = self.dec(z)
                return outs

        self.model = _AE(self.encoder, self.decoder).to(self.device)
        return self.model

    def _make_loader(self) -> DataLoader:
        ds = _RowDataset(self.row_generator, self.fields)
        bs = self.config.batch_size
        return DataLoader(
            ds,
            batch_size=bs,
            collate_fn=_collate,
        )

    def save_model(self):
        out = Path(self.model_dir)
        try:
            out.mkdir(parents=True, exist_ok=True)
            name = self.__class__.__name__
            torch.save(
                self.model.state_dict(),
                out / f"{name}_autoencoder.pt",
            )
            torch.save(
                self.encoder.state_dict(),
                out / f"{name}_encoder.pt",
            )
            torch.save(
                self.decoder.state_dict(),
                out / f"{name}_decoder.pt",
            )
        except Exception as e:
            logging.error(f"Failed to save model {self.__class__.__name__} to {out}: {e}")
            logging.error("Continuing training without saving this checkpoint.")

    def fit(self):
        if not self.stats_accumulated:
            self.accumulate_stats()
            self.finalize_stats()
        if self.model is None:
            self.build_autoencoder()

        epochs = self.config.epochs
        lr = self.config.learning_rate
        wd = self.config.weight_decay

        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=wd,
        )

        loader = self._make_loader()

        self.model.train()
        global_step = 0
        for epoch in range(epochs):
            pbar = tqdm(
                loader,
                desc=f"{self.__class__.__name__} epoch {epoch+1}/{epochs}",
            )
            for xs, ys in pbar:
                xs = [x.to(self.device) for x in xs]
                ys = [y.to(self.device) for y in ys]

                outs = self.model(xs)

                total = 0.0
                for f, pred, tgt in zip(
                    self.fields,
                    outs,
                    ys,
                ):
                    loss = f.compute_loss(pred, tgt) * float(
                        f.weight
                    )
                    total = total + loss

                opt.zero_grad()
                total.backward()
                opt.step()

                global_step += 1
                pbar.set_postfix(
                    loss=float(
                        total.detach().cpu().item()
                    )
                )

            self.save_model()
        return None