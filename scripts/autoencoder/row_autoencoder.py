import logging
import pickle
import sqlite3
from pathlib import Path
from typing import Any, List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

from autoencoder.fields import BaseField

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class _FieldEncoders(nn.Module):
    def __init__(self, fields: List[BaseField], latent_dim: int):
        super().__init__()
        self.fields = fields
        self.encs = nn.ModuleList([f.build_encoder(latent_dim) for f in fields])
        self.proj = nn.ModuleList([nn.Identity() if _out_dim(m) == latent_dim else nn.Linear(_out_dim(m), latent_dim) for m in self.encs])
        self.fuse = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        outs = []
        for x, enc, proj in zip(xs, self.encs, self.proj):
            y = enc(x)
            if y.dim() > 2:
                y = y.flatten(1)
            y = proj(y)
            outs.append(y)
        tokens = torch.stack(outs, dim=1)
        attn_out, _ = self.fuse(tokens, tokens, tokens)
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


class RowAutoencoder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_dir = Path(config["model_dir"])
        self.model = None
        self.encoder = None
        self.decoder = None

        self.fields: List[BaseField] = self.build_fields()
        self.latent_dim = int(self.config["latent_dim"])
        self.num_rows_in_dataset = 0

        self.db_path: str = config["db_path"]
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
                blob = pickle.dumps(f)
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
            cache_map = {name: pickle.loads(blob) for name, blob in rows}
            for i, f in enumerate(self.fields):
                if f.name in cache_map:
                    self.fields[i] = cache_map[f.name]
        self.stats_accumulated = True
        return True

    def accumulate_stats(self):
        use_cache = self.config.get("use_cache", True)
        refresh_cache = self.config.get("refresh_cache", False)
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
        bs = int(self.config.get("batch_size", 32))
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

        epochs = int(self.config.get("epochs", 10))
        lr = float(self.config.get("learning_rate", 2e-4))
        wd = float(self.config.get("weight_decay", 1e-4))

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
