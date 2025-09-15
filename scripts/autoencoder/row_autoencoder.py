# scripts/autoencoder/row_autoencoder.py

import logging
import json
import sqlite3
from pathlib import Path
from typing import Any, List, Dict, Tuple

import numpy as np
from config import ProjectConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

from .fields import (
    BaseField,
    ScalarField,
    BooleanField,
    MultiCategoryField,
    SingleCategoryField,
    TextField,
    NumericDigitCategoryField,
)
from .character_tokenizer import CharacterTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class _FieldEncoders(nn.Module):
    def __init__(self, fields: List[BaseField], latent_dim: int):
        super().__init__()
        self.fields = fields
        self.encs = nn.ModuleList([f.build_encoder(latent_dim) for f in fields])
        self.proj = nn.ModuleList([nn.Identity() if _out_dim(m) == latent_dim else nn.Linear(_out_dim(m), latent_dim) for m in self.encs])
        self.fuse = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(latent_dim)
        self.field_embed = nn.Parameter(torch.randn(len(fields), latent_dim) * 0.02)
        self.q_bias = nn.Parameter(torch.zeros(len(fields), latent_dim))
        self.k_bias = nn.Parameter(torch.zeros(len(fields), latent_dim))

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        outs = []
        for x, enc, proj in zip(xs, self.encs, self.proj):
            y = enc(x)
            if y.dim() > 2:
                y = y.flatten(1)
            y = proj(y)
            outs.append(y)
        tokens = torch.stack(outs, dim=1)
        tokens = tokens + self.field_embed.unsqueeze(0)
        q = tokens + self.q_bias.unsqueeze(0)
        k = tokens + self.k_bias.unsqueeze(0)
        attn_out, _ = self.fuse(q, k, tokens)
        z = self.norm(attn_out.mean(dim=1))
        return z



class _FieldDecoders(nn.Module):
    def __init__(self, fields: List[BaseField], latent_dim: int):
        super().__init__()
        self.fields = fields
        self.decs = nn.ModuleList([f.build_decoder(latent_dim) for f in fields])

    def forward(self, z: torch.Tensor) -> List[torch.Tensor]:
        outs = []
        for dec in self.decs:
            y = dec(z)
            outs.append(y)
        return outs


def _out_dim(m: nn.Module) -> int:
    for p in reversed(list(m.parameters())):
        if p.dim() == 2:
            return p.size(0)
    return None


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


def _collate(batch: List[Tuple[List[torch.Tensor], List[torch.Tensor]]]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    x_cols = list(zip(*[b[0] for b in batch]))
    y_cols = list(zip(*[b[1] for b in batch]))
    X = [torch.stack(col, dim=0) for col in x_cols]
    Y = [torch.stack(col, dim=0) for col in y_cols]
    return X, Y


def _field_to_state(f: BaseField) -> Dict[str, Any]:
    state: Dict[str, Any] = {"name": f.name, "optional": bool(getattr(f, "optional", False)), "type": f.__class__.__name__}

    if isinstance(f, BooleanField):
        state.update({
            "use_bce_loss": bool(getattr(f, "use_bce_loss", True)),
            "count_total": int(getattr(f, "count_total", 0)),
            "count_ones": int(getattr(f, "count_ones", 0)),
        })
        return state

    if isinstance(f, ScalarField):
        state.update({
            "scaling": int(getattr(f, "scaling", None).value if getattr(f, "scaling", None) is not None else 1),
            "clip_max": getattr(f, "clip_max", None),
            "n": int(getattr(f, "n", 0)),
            "sum_": float(getattr(f, "sum_", 0.0)),
            "sum_sq": float(getattr(f, "sum_sq", 0.0)),
            "min_val": float(getattr(f, "min_val", 0.0) if np.isfinite(getattr(f, "min_val", 0.0)) else 0.0),
            "max_val": float(getattr(f, "max_val", 0.0) if np.isfinite(getattr(f, "max_val", 0.0)) else 0.0),
            "mean_val": float(getattr(f, "mean_val", 0.0)),
            "std_val": float(getattr(f, "std_val", 1.0)),
        })
        return state

    if isinstance(f, MultiCategoryField):
        state.update({
            "category_list": list(getattr(f, "category_list", []) or []),
            "category_counts": dict(getattr(f, "category_counts", {}) or {}),
        })
        return state

    if isinstance(f, SingleCategoryField):
        state.update({
            "category_list": list(getattr(f, "category_list", []) or []),
            "category_counts": dict(getattr(f, "category_counts", {}) or {}),
        })
        return state

    if isinstance(f, NumericDigitCategoryField):
        state.update({
            "base": int(getattr(f, "base", 10)),
            "fraction_digits": int(getattr(f, "fraction_digits", 0)),
            "has_negative": bool(getattr(f, "has_negative", False)),
            "has_nan": bool(getattr(f, "has_nan", False)),
            "integer_digits": int(getattr(f, "integer_digits", 1) if getattr(f, "integer_digits", None) is not None else 1),
            "total_positions": int(getattr(f, "total_positions", 1) if getattr(f, "total_positions", None) is not None else 1),
        })
        return state

    if isinstance(f, TextField):
        tok = getattr(f, "tokenizer", None)
        vocab = None
        specials = None
        if tok is not None and getattr(tok, "char_to_index", None):
            max_id = max(tok.index_to_char.keys()) if tok.index_to_char else -1
            vocab = [tok.index_to_char.get(i, "") for i in range(max_id + 1)]
            specials = list(getattr(tok, "special_tokens", []) or [])
        state.update({
            "user_max_length": getattr(f, "user_max_length", None),
            "downsample_steps": int(getattr(f, "downsample_steps", 2)),
            "base_size": int(getattr(f, "base_size", 48)),
            "num_blocks_per_step": list(getattr(f, "num_blocks_per_step", [2, 2])),
            "dynamic_max_len": int(getattr(f, "dynamic_max_len", 0)),
            "max_length": int(getattr(f, "max_length", 1) if getattr(f, "max_length", None) is not None else 1),
            "pad_token_id": int(getattr(f, "pad_token_id", 0) if getattr(f, "pad_token_id", None) is not None else 0),
            "avg_raw_length": float(getattr(f, "avg_raw_length", 0.0) or 0.0),
            "avg_token_count": float(getattr(f, "avg_token_count", 0.0) or 0.0),
            "avg_chars_saved": float(getattr(f, "avg_chars_saved", 0.0) or 0.0),
            "compression_ratio": float(getattr(f, "compression_ratio", 0.0) or 0.0),
            "tokenizer_vocab": vocab,
            "tokenizer_specials": specials,
        })
        return state

    return state


def _apply_field_state(f: BaseField, st: Dict[str, Any]) -> None:
    if st.get("name") and f.name != st["name"]:
        return

    f.optional = bool(st.get("optional", getattr(f, "optional", False)))

    if isinstance(f, BooleanField):
        f.use_bce_loss = bool(st.get("use_bce_loss", getattr(f, "use_bce_loss", True)))
        f.count_total = int(st.get("count_total", 0))
        f.count_ones = int(st.get("count_ones", 0))
        return

    if isinstance(f, ScalarField):
        f.scaling = f.scaling.__class__(int(st.get("scaling", 1)))
        f.clip_max = st.get("clip_max", None)
        f.n = int(st.get("n", 0))
        f.sum_ = float(st.get("sum_", 0.0))
        f.sum_sq = float(st.get("sum_sq", 0.0))
        f.min_val = float(st.get("min_val", 0.0))
        f.max_val = float(st.get("max_val", 0.0))
        f.mean_val = float(st.get("mean_val", 0.0))
        f.std_val = float(st.get("std_val", 1.0))
        return

    if isinstance(f, MultiCategoryField):
        f.category_list = list(st.get("category_list", []) or [])
        f.category_set = set(f.category_list)
        f.category_counts = dict(st.get("category_counts", {}) or {})
        return

    if isinstance(f, SingleCategoryField):
        f.category_list = list(st.get("category_list", []) or [])
        f.category_set = set(f.category_list)
        f.category_counts = dict(st.get("category_counts", {}) or {})
        return

    if isinstance(f, NumericDigitCategoryField):
        f.base = int(st.get("base", getattr(f, "base", 10)))
        f.fraction_digits = int(st.get("fraction_digits", getattr(f, "fraction_digits", 0)))
        f.has_negative = bool(st.get("has_negative", getattr(f, "has_negative", False)))
        f.has_nan = bool(st.get("has_nan", getattr(f, "has_nan", False)))
        f.integer_digits = int(st.get("integer_digits", getattr(f, "integer_digits", 1)))
        f.total_positions = int(st.get("total_positions", getattr(f, "total_positions", f.integer_digits)))
        return

    if isinstance(f, TextField):
        f.user_max_length = st.get("user_max_length", getattr(f, "user_max_length", None))
        f.downsample_steps = int(st.get("downsample_steps", getattr(f, "downsample_steps", 2)))
        f.base_size = int(st.get("base_size", getattr(f, "base_size", 48)))
        f.num_blocks_per_step = list(st.get("num_blocks_per_step", getattr(f, "num_blocks_per_step", [2, 2])))
        f.dynamic_max_len = int(st.get("dynamic_max_len", getattr(f, "dynamic_max_len", 0)))
        f.max_length = int(st.get("max_length", getattr(f, "max_length", 1)))
        f.pad_token_id = int(st.get("pad_token_id", getattr(f, "pad_token_id", 0)))
        f.avg_raw_length = float(st.get("avg_raw_length", getattr(f, "avg_raw_length", 0.0)) or 0.0)
        f.avg_token_count = float(st.get("avg_token_count", getattr(f, "avg_token_count", 0.0)) or 0.0)
        f.avg_chars_saved = float(st.get("avg_chars_saved", getattr(f, "avg_chars_saved", 0.0)) or 0.0)
        f.compression_ratio = float(st.get("compression_ratio", getattr(f, "compression_ratio", 0.0)) or 0.0)

        vocab = st.get("tokenizer_vocab", None)
        specials = st.get("tokenizer_specials", None)
        if vocab is not None:
            tok = CharacterTokenizer(special_tokens=specials or [])
            tok.char_to_index = {ch: i for i, ch in enumerate(vocab)}
            tok.index_to_char = {i: ch for i, ch in enumerate(vocab)}
            tok.alphabet = set([c for c in vocab if c not in (specials or [])])
            tok.trained = True
            f.tokenizer = tok
        return


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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _cache_table_name(self) -> str:
        return f"{self.__class__.__name__}_stats_cache"

    def _drop_cache_table(self):
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            conn.execute(f"DROP TABLE IF EXISTS {self._cache_table_name()};")
            conn.commit()

    def _save_cache(self):
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self._cache_table_name()} (field_name TEXT PRIMARY KEY, data BLOB)"
            )
            for f in self.fields:
                st = _field_to_state(f)
                blob = json.dumps(st, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                conn.execute(
                    f"INSERT OR REPLACE INTO {self._cache_table_name()} (field_name, data) VALUES (?, ?);",
                    (f.name, blob),
                )
            conn.commit()

    def _load_cache(self) -> bool:
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (self._cache_table_name(),)
            )
            if cur.fetchone() is None:
                return False
            cur.execute(f"SELECT field_name, data FROM {self._cache_table_name()};")
            rows = cur.fetchall()
            if not rows:
                return False
            cache_map: Dict[str, Dict[str, Any]] = {}
            for name, blob in rows:
                try:
                    cache_map[name] = json.loads(blob.decode("utf-8"))
                except Exception:
                    return False
            for i, f in enumerate(self.fields):
                st = cache_map.get(f.name)
                if st:
                    _apply_field_state(f, st)
        self.stats_accumulated = True
        return True

    def accumulate_stats(self):
        use_cache = self.config.use_cache
        refresh_cache = self.config.refresh_cache
        if refresh_cache:
            self._drop_cache_table()
        if use_cache and self._load_cache():
            logging.info("stats loaded from cache")
            return
        if self.stats_accumulated:
            logging.info("stats already accumulated")
            return
        n = 0
        logging.info("accumulating stats")
        for row in tqdm(self.row_generator(), desc=self.__class__.__name__):
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
        z = torch.tensor(latent_vector, dtype=torch.float32, device=self.device)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        with torch.no_grad():
            outs = self.decoder(z)
        rec = {}
        for f, o in zip(self.fields, outs):
            arr = o.detach().cpu().numpy()[0]
            rec[f.name] = f.to_string(arr)
        return rec

    def build_fields(self) -> List["BaseField"]:
        raise NotImplementedError

    def row_generator(self):
        raise NotImplementedError

    def build_autoencoder(self):
        if not self.stats_accumulated:
            raise RuntimeError("Call accumulate_stats()/finalize_stats() first")

        self.encoder = _FieldEncoders(self.fields, self.latent_dim).to(self.device)
        self.decoder = _FieldDecoders(self.fields, self.latent_dim).to(self.device)

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
        return DataLoader(ds, batch_size=bs, collate_fn=_collate)

    def save_model(self):
        out = Path(self.model_dir)
        out.mkdir(parents=True, exist_ok=True)
        name = self.__class__.__name__
        torch.save(self.model.state_dict(), out / f"{name}_autoencoder.pt")
        torch.save(self.encoder.state_dict(), out / f"{name}_encoder.pt")
        torch.save(self.decoder.state_dict(), out / f"{name}_decoder.pt")

    def fit(self):
        if not self.stats_accumulated:
            self.accumulate_stats()
            self.finalize_stats()
        if self.model is None:
            self.build_autoencoder()

        epochs = self.config.epochs
        lr = self.config.learning_rate
        wd = self.config.weight_decay

        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)

        loader = self._make_loader()

        self.model.train()
        global_step = 0
        for epoch in range(epochs):
            pbar = tqdm(loader, desc=f"{self.__class__.__name__} epoch {epoch+1}/{epochs}")
            for xs, ys in pbar:
                xs = [x.to(self.device) for x in xs]
                ys = [y.to(self.device) for y in ys]

                outs = self.model(xs)

                total = 0.0
                for f, pred, tgt in zip(self.fields, outs, ys):
                    loss = f.compute_loss(pred, tgt) * float(f.weight)
                    total = total + loss

                opt.zero_grad()
                total.backward()
                opt.step()

                global_step += 1
                pbar.set_postfix(loss=float(total.detach().cpu().item()))

            self.save_model()
        return None
