# scripts/train_imdb_sequence_decoder.py

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sqlite3
from scripts.autoencoder.print_model import print_model_layers_with_shapes
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

from config import project_config
from scripts.autoencoder.imdb_row_autoencoders import TitlesAutoencoder, PeopleAutoencoder
from scripts.autoencoder.fields import BaseField
from scripts.autoencoder.fields import TextField, MultiCategoryField, BooleanField, ScalarField, SingleCategoryField, NumericDigitCategoryField
from scripts.autoencoder.training_callbacks import SequenceReconstructionLogger
from scripts.autoencoder.run_logger import build_run_logger

class _GPUEventTimer:
    def __init__(self, print_every: int):
        self.print_every = max(1, int(print_every))
        self.reset_accum()
        self._step = 0

    def reset_accum(self):
        self.accum_ms = {
            "total": 0.0,
            "data": 0.0,
            "h2d": 0.0,
            "mov_enc": 0.0,
            "trunk": 0.0,
            "ppl_dec": 0.0,
            "rec": 0.0,
            "tgt_enc": 0.0,
            "nce": 0.0,
            "backward": 0.0,
            "opt": 0.0,
        }

    def _event_pair(self):
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def start_step(self):
        self._t0 = time.perf_counter()
        self._pairs = {}

    def end_step_and_accumulate(self):
        self.accum_ms["total"] += (time.perf_counter() - self._t0) * 1000.0
        self._step += 1
        if self._step % self.print_every != 0:
            return None
        torch.cuda.synchronize()
        out = {}
        total = self.accum_ms["total"]
        parts = 0.0
        for k, v in self.accum_ms.items():
            if k == "total":
                continue
            parts += v
        residual = max(0.0, total - parts)
        keys = ["total","backward","trunk","mov_enc","tgt_enc","rec","opt","ppl_dec","nce","data","h2d","residual"]
        out["total"] = total
        out["backward"] = self.accum_ms["backward"]
        out["trunk"] = self.accum_ms["trunk"]
        out["mov_enc"] = self.accum_ms["mov_enc"]
        out["tgt_enc"] = self.accum_ms["tgt_enc"]
        out["rec"] = self.accum_ms["rec"]
        out["opt"] = self.accum_ms["opt"]
        out["ppl_dec"] = self.accum_ms["ppl_dec"]
        out["nce"] = self.accum_ms["nce"]
        out["data"] = self.accum_ms["data"]
        out["h2d"] = self.accum_ms["h2d"]
        out["residual"] = residual
        self.reset_accum()
        return keys, out

    def cpu_range(self, name):
        class _R:
            def __init__(self, outer, nm):
                self.o = outer
                self.nm = nm
            def __enter__(self):
                self.t0 = time.perf_counter()
            def __exit__(self, a, b, c):
                self.o.accum_ms[self.nm] += (time.perf_counter() - self.t0) * 1000.0
        return _R(self, name)

    def gpu_range(self, name):
        class _R:
            def __init__(self, outer, nm):
                self.o = outer
                self.nm = nm
            def __enter__(self):
                self.s, self.e = self.o._event_pair()
                self.s.record()
            def __exit__(self, a, b, c):
                self.e.record()
                torch.cuda.synchronize()
                self.o.accum_ms[self.nm] += self.s.elapsed_time(self.e)
        return _R(self, name)

    def print_line(self, keys, vals, step_idx):
        total = vals["total"]
        frags = []
        for k in keys:
            ms = vals[k]
            pct = 0.0 if total <= 0 else (ms / total) * 100.0
            frags.append(f"{k:>8}: {ms:7.2f} ms {pct:6.1f}%")
        print(f"[step {step_idx}] " + " | ".join(frags))

class CudaPrefetcher:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None
        self.it = iter(loader)
        self.next_batch = None
        self._preload()

    def _to_device(self, batch):
        xm, yp, m = batch
        xm = [x.to(self.device, non_blocking=True) for x in xm]
        yp = [y.to(self.device, non_blocking=True) for y in yp]
        m = m.to(self.device, non_blocking=True)
        return xm, yp, m

    def _preload(self):
        try:
            batch = next(self.it)
        except StopIteration:
            self.next_batch = None
            return
        if self.stream is None:
            self.next_batch = self._to_device(batch)
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = self._to_device(batch)

    def next(self):
        if self.next_batch is None:
            return None
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self._preload()
        return batch

class MoviesPeopleSequenceDataset(IterableDataset):
    def __init__(
        self,
        db_path: str,
        movie_fields: List[BaseField],
        people_fields: List[BaseField],
        seq_len: int,
        movie_limit: int | None = None,
    ):
        super().__init__()
        self.db_path = db_path
        self.movie_fields = movie_fields
        self.people_fields = people_fields
        self.seq_len = seq_len
        self.movie_limit = movie_limit
        self.movie_sql = """
        SELECT
            t.tconst,
            t.primaryTitle,
            t.startYear,
            t.endYear,
            t.runtimeMinutes,
            t.averageRating,
            t.numVotes,
            GROUP_CONCAT(g.genre, ',')
        FROM titles t
        INNER JOIN title_genres g ON g.tconst = t.tconst
        WHERE
            t.startYear IS NOT NULL
            AND t.averageRating IS NOT NULL
            AND t.runtimeMinutes IS NOT NULL
            AND t.runtimeMinutes >= 5
            AND t.startYear >= 1850
            AND t.titleType IN ('movie','tvSeries','tvMovie','tvMiniSeries')
            AND t.numVotes >= 10
        GROUP BY t.tconst
        """
        self.seq_lookup_sql = "SELECT people_json FROM movie_people_seq WHERE tconst = ? LIMIT 1"

    def __iter__(self):
        import json
        import sqlite3
        from torch.utils.data import get_worker_info

        con = sqlite3.connect(self.db_path, check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute("PRAGMA temp_store=MEMORY;")
        con.execute("PRAGMA cache_size=-200000;")
        con.execute("PRAGMA mmap_size=268435456;")
        con.execute("PRAGMA busy_timeout=5000;")

        cur = con.cursor()
        lim = "" if self.movie_limit is None else f" LIMIT {int(self.movie_limit)}"
        cur.execute(self.movie_sql + lim)

        wi = get_worker_info()
        if wi is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = wi.id
            num_workers = wi.num_workers

        for idx, row in enumerate(cur):
            if (idx % num_workers) != worker_id:
                continue

            tconst, title, startYear, endYear, runtime, rating, votes, genres = row
            s = con.execute(self.seq_lookup_sql, (tconst,)).fetchone()
            if not s:
                continue
            ppl = json.loads(s[0]) or []
            orig_len = min(len(ppl), self.seq_len)
            if orig_len < self.seq_len:
                ppl = ppl + [{} for _ in range(self.seq_len - orig_len)]
            else:
                ppl = ppl[: self.seq_len]

            movie_row = {
                "tconst": tconst,
                "primaryTitle": title,
                "startYear": startYear,
                "endYear": endYear,
                "runtimeMinutes": runtime,
                "averageRating": rating,
                "numVotes": votes,
                "genres": genres.split(",") if genres else [],
            }

            x_movie = [f.transform(movie_row.get(f.name)) for f in self.movie_fields]

            y_people = []
            for f in self.people_fields:
                steps = []
                for t in range(self.seq_len):
                    if t < orig_len:
                        steps.append(f.transform_target(ppl[t].get(f.name)))
                    else:
                        steps.append(f.get_base_padding_value())
                y_people.append(torch.stack(steps, dim=0))

            mask = torch.zeros(self.seq_len, dtype=torch.float32)
            mask[:orig_len] = 1.0
            yield x_movie, y_people, mask

        con.close()


def _collate(batch):
    xm_cols = list(zip(*[b[0] for b in batch]))
    yp_cols = list(zip(*[b[1] for b in batch]))
    Xm = [torch.stack(col, dim=0) for col in xm_cols]
    Yp = [torch.stack(col, dim=0) for col in yp_cols]
    M = torch.stack([b[2] for b in batch], dim=0)
    return Xm, Yp, M


class _TransformerTrunk(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        seq_len: int,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.query = nn.Parameter(torch.randn(seq_len, latent_dim))
        layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_mult * latent_dim,
            dropout=dropout,
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, z_m: torch.Tensor) -> torch.Tensor:
        memory = z_m.unsqueeze(0)
        q = self.query.unsqueeze(1).expand(self.seq_len, z_m.size(0), z_m.size(1))
        out = self.decoder(q, memory)
        out = out.transpose(0, 1)
        out = self.norm(out)
        return out


class MovieToPeopleSequencePredictor(nn.Module):
    def __init__(
        self,
        movie_encoder: nn.Module,
        people_decoder: nn.Module,
        latent_dim: int,
        seq_len: int,
        width: int | None = None,
        depth: int = 3,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.movie_encoder = movie_encoder
        self.people_decoder = people_decoder
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.trunk = _TransformerTrunk(
            latent_dim=latent_dim,
            seq_len=seq_len,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_mult=ff_mult,
            dropout=dropout,
        )

    def forward(self, movie_inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        z_m = self.movie_encoder(movie_inputs)
        z_seq = self.trunk(z_m)
        b = z_seq.size(0)
        flat = z_seq.reshape(b * self.seq_len, self.latent_dim)
        outs = self.people_decoder(flat)
        seq_outs = []
        for y in outs:
            seq_outs.append(y.view(b, self.seq_len, *y.shape[1:]))
        return seq_outs


def _per_sample_field_loss_seq(field: BaseField, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    if isinstance(field, ScalarField):
        diff = pred - tgt
        if diff.dim() > 2:
            diff = diff.flatten(2)
            loss = diff.pow(2).mean(dim=2)
        else:
            loss = diff.pow(2)
        return loss
    if isinstance(field, BooleanField):
        if getattr(field, "use_bce_loss", True):
            loss = F.binary_cross_entropy_with_logits(pred, tgt, reduction="none")
            if loss.dim() > 2:
                loss = loss.flatten(2).mean(dim=2)
            return loss
        diff = torch.tanh(pred) - tgt
        if diff.dim() > 2:
            diff = diff.flatten(2).mean(dim=2)
        else:
            diff = diff
        return diff.pow(2)
    if isinstance(field, MultiCategoryField):
        loss = F.binary_cross_entropy_with_logits(pred, tgt, reduction="none")
        if loss.dim() > 2:
            loss = loss.flatten(2).mean(dim=2)
        return loss
    if isinstance(field, SingleCategoryField):
        B, T = pred.shape[0], pred.shape[1]
        C = pred.shape[-1]
        t = tgt.long().squeeze(-1)
        loss = F.cross_entropy(pred.view(B * T, C), t.view(B * T), reduction="none")
        return loss.view(B, T)
    if isinstance(field, TextField):
        B, T, L, V = pred.shape
        pad_id = int(field.pad_token_id)
        loss_flat = F.cross_entropy(pred.view(B * T, L, V).reshape(B * T * L, V), tgt.view(B * T, L).reshape(B * T * L), ignore_index=pad_id, reduction="none")
        loss_flat = loss_flat.view(B * T, L)
        mask_tok = (tgt.view(B * T, L) != pad_id).float()
        denom = mask_tok.sum(dim=1).clamp_min(1.0)
        loss_bt = (loss_flat * mask_tok).sum(dim=1) / denom
        return loss_bt.view(B, T)
    if isinstance(field, NumericDigitCategoryField):
        B, T, P, V = pred.shape
        loss_flat = F.cross_entropy(pred.view(B * T * P, V), tgt.view(B * T * P).long(), reduction="none")
        loss_bt = loss_flat.view(B * T, P).mean(dim=1)
        return loss_bt.view(B, T)
    diff = pred - tgt
    if diff.dim() > 2:
        diff = diff.flatten(2).mean(dim=2)
    return diff.pow(2)


def _sequence_loss_and_breakdown(
    fields: List[BaseField],
    preds_seq: List[torch.Tensor],
    targets_seq: List[torch.Tensor],
    mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    total = 0.0
    field_losses: dict[str, float] = {}
    denom = mask.sum().clamp_min(1.0)
    for f, pred, tgt in zip(fields, preds_seq, targets_seq):
        per_bt = _per_sample_field_loss_seq(f, pred, tgt)
        wsum = (per_bt * mask).sum()
        val = wsum / denom
        total = total + val * float(f.weight)
        field_losses[f.name] = float(val.detach().cpu().item())
    return total, field_losses


def _unit_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


def _info_nce_pair(z_pred: torch.Tensor, z_tgt: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = z_pred @ z_tgt.t()
    logits = logits / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss = F.cross_entropy(logits, labels)
    return loss


def _encode_people_latents_vectorized(
    people_encoder: nn.Module,
    targets_seq: List[torch.Tensor],
    timestep: int,
    indices: torch.Tensor,
) -> torch.Tensor:
    xs = []
    for y in targets_seq:
        sl = y.narrow(1, timestep, 1).squeeze(1)
        sl = sl.index_select(0, indices)
        xs.append(sl)
    with torch.no_grad():
        z = people_encoder(xs)
    return z



def _load_frozen_autoencoders(config: Dict[str, Any]) -> Tuple[TitlesAutoencoder, PeopleAutoencoder]:
    mov = TitlesAutoencoder(config)
    per = PeopleAutoencoder(config)
    mov.accumulate_stats()
    mov.finalize_stats()
    per.accumulate_stats()
    per.finalize_stats()
    mov.build_autoencoder()
    per.build_autoencoder()
    model_dir = Path(config["model_dir"])
    mov.encoder.load_state_dict(torch.load(model_dir / "TitlesAutoencoder_encoder.pt", map_location="cpu"))
    per.decoder.load_state_dict(torch.load(model_dir / "PeopleAutoencoder_decoder.pt", map_location="cpu"))
    per.encoder.load_state_dict(torch.load(model_dir / "PeopleAutoencoder_encoder.pt", map_location="cpu"))
    for p in mov.encoder.parameters():
        p.requires_grad = False
    for p in per.decoder.parameters():
        p.requires_grad = False
    for p in per.encoder.parameters():
        p.requires_grad = False
    mov.encoder.eval()
    per.decoder.eval()
    per.encoder.eval()
    return mov, per

class MoviesPeopleSequenceMemoryDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        db_path: str,
        movie_fields: List[BaseField],
        people_fields: List[BaseField],
        seq_len: int,
        movie_limit: int | None = None,
    ):
        super().__init__()
        self.db_path = db_path
        self.movie_fields = movie_fields
        self.people_fields = people_fields
        self.seq_len = seq_len
        self.movie_limit = movie_limit
        self.movie_sql = """
        SELECT
            t.tconst,
            t.primaryTitle,
            t.startYear,
            t.endYear,
            t.runtimeMinutes,
            t.averageRating,
            t.numVotes,
            GROUP_CONCAT(g.genre, ',')
        FROM titles t
        INNER JOIN title_genres g ON g.tconst = t.tconst
        WHERE
            t.startYear IS NOT NULL
            AND t.averageRating IS NOT NULL
            AND t.runtimeMinutes IS NOT NULL
            AND t.runtimeMinutes >= 5
            AND t.startYear >= 1850
            AND t.titleType IN ('movie','tvSeries','tvMovie','tvMiniSeries')
            AND t.numVotes >= 10
        GROUP BY t.tconst
        """
        self.people_sql = """
        SELECT
            p.primaryName,
            p.birthYear,
            p.deathYear,
            GROUP_CONCAT(pp.profession, ',')
        FROM people p
        LEFT JOIN people_professions pp ON p.nconst = pp.nconst
        INNER JOIN principals pr ON pr.nconst = p.nconst
        WHERE pr.tconst = ? AND p.birthYear IS NOT NULL
        GROUP BY p.nconst
        HAVING COUNT(pp.profession) > 0
        ORDER BY pr.ordering
        LIMIT ?
        """
        self.samples = self._materialize()

    def _materialize(self):
        import sqlite3
        con = sqlite3.connect(self.db_path, check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute("PRAGMA temp_store=MEMORY;")
        con.execute("PRAGMA cache_size=-200000;")
        con.execute("PRAGMA mmap_size=268435456;")
        con.execute("PRAGMA busy_timeout=5000;")
        cur = con.cursor()
        lim = "" if self.movie_limit is None else f" LIMIT {int(self.movie_limit)}"
        cur.execute(self.movie_sql + lim)
        out = []
        for tconst, title, startYear, endYear, runtime, rating, votes, genres in cur:
            movie_row = {
                "tconst": tconst,
                "primaryTitle": title,
                "startYear": startYear,
                "endYear": endYear,
                "runtimeMinutes": runtime,
                "averageRating": rating,
                "numVotes": votes,
                "genres": genres.split(",") if genres else [],
            }
            ppl_rows = con.execute(self.people_sql, (tconst, self.seq_len)).fetchall()
            if not ppl_rows:
                continue
            ppl = []
            for r in ppl_rows:
                ppl.append({
                    "primaryName": r[0],
                    "birthYear": r[1],
                    "deathYear": r[2],
                    "professions": r[3].split(",") if r[3] else None,
                })
            orig_len = min(len(ppl), self.seq_len)
            if orig_len < self.seq_len:
                ppl = ppl + [{} for _ in range(self.seq_len - orig_len)]
            else:
                ppl = ppl[: self.seq_len]
            x_movie = [f.transform(movie_row.get(f.name)) for f in self.movie_fields]
            y_people = []
            for f in self.people_fields:
                steps = []
                for t in range(self.seq_len):
                    if t < orig_len:
                        steps.append(f.transform_target(ppl[t].get(f.name)))
                    else:
                        steps.append(f.get_base_padding_value())
                y_people.append(torch.stack(steps, dim=0))
            mask = torch.zeros(self.seq_len, dtype=torch.float32)
            mask[:orig_len] = 1.0
            out.append((x_movie, y_people, mask))
        con.close()
        return out

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]

class CudaPrefetcher:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None
        self.it = iter(loader)
        self.next_batch = None
        self._preload()

    def _to_device(self, batch):
        xm, yp, m = batch
        xm = [x.to(self.device, non_blocking=True) for x in xm]
        yp = [y.to(self.device, non_blocking=True) for y in yp]
        m = m.to(self.device, non_blocking=True)
        return xm, yp, m

    def _preload(self):
        try:
            batch = next(self.it)
        except StopIteration:
            self.next_batch = None
            return
        if self.stream is None:
            self.next_batch = self._to_device(batch)
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = self._to_device(batch)

    def next(self):
        if self.next_batch is None:
            return None
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self._preload()
        return batch

def _info_nce_masked_rows(z_pred_seq: torch.Tensor, z_tgt_seq: torch.Tensor, mask: torch.Tensor, temperature: float) -> torch.Tensor:
    z_pred_seq = F.normalize(z_pred_seq, dim=-1)
    z_tgt_seq = F.normalize(z_tgt_seq, dim=-1)
    B, T, _ = z_pred_seq.shape
    losses = []
    for t in range(T):
        row_mask = mask[:, t] > 0.5
        k = int(row_mask.sum().item())
        if k < 2:
            continue
        zp = z_pred_seq[row_mask, t, :]
        zt = z_tgt_seq[row_mask, t, :]
        logits = (zp @ zt.t()) / temperature
        labels = torch.arange(k, device=logits.device)
        losses.append(F.cross_entropy(logits, labels))
    if losses:
        return torch.stack(losses).mean()
    return z_pred_seq.new_zeros(())

def disable_inductor_autotune():
    from torch._inductor import config as _inductor_cfg
    _inductor_cfg.max_autotune = False


def train_sequence_predictor(
    config: Dict[str, Any],
    steps: int,
    save_every: int,
    movie_limit: int | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mov, per = _load_frozen_autoencoders(config)

    latent_dim = int(config["latent_dim"])
    seq_len = int(config["people_sequence_length"])
    batch_size = int(config["batch_size"])
    lr = float(config.get("learning_rate", 2e-4))
    wd = float(config.get("weight_decay", 1e-4))
    temp = float(config.get("latent_temperature", 0.07))
    w_lat = float(config.get("latent_loss_weight", 1.0))
    w_rec = float(config.get("recon_loss_weight", 0.1))
    log_every = int(config.get("log_interval", 20))
    use_compile = bool(config.get("compile_trunk", True))
    use_cuda_graphs = bool(config.get("use_cuda_graphs", False))

    disable_inductor_autotune()
    from scripts.precompute_movie_people_seq import build_movie_people_seq
    import sqlite3
    _conn = sqlite3.connect(str(Path(config["db_path"])))
    _has = _conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='movie_people_seq'").fetchone()
    _conn.close()
    if _has is None:
        build_movie_people_seq(str(Path(config["db_path"])), seq_len)

    model = MovieToPeopleSequencePredictor(
        movie_encoder=mov.encoder,
        people_decoder=per.decoder,
        latent_dim=latent_dim,
        seq_len=seq_len,
    ).to(device)

    if use_compile and hasattr(torch, "compile"):
        try:
            model.trunk = torch.compile(model.trunk, mode="max-autotune")
        except Exception:
            pass

    ds = MoviesPeopleSequenceDataset(
        db_path=str(Path(config["db_path"])),
        movie_fields=mov.fields,
        people_fields=per.fields,
        seq_len=seq_len,
        movie_limit=movie_limit,
    )

    num_workers = int(config.get("num_workers", 2))
    prefetch_factor = int(config.get("prefetch_factor", 2))
    pin = bool(torch.cuda.is_available())

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=_collate,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else 2,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=pin,
        drop_last=False,
    )

    mov.encoder.to(device)
    per.decoder.to(device)
    per.encoder.to(device)

    opt = torch.optim.AdamW(model.trunk.parameters(), lr=lr, weight_decay=wd, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    run_logger = build_run_logger(config)
    seq_logger = SequenceReconstructionLogger(
        movie_ae=mov,
        people_ae=per,
        predictor=model,
        db_path=str(Path(config["db_path"])),
        seq_len=seq_len,
        interval_steps=int(config.get("callback_interval", 100)),
        num_samples=2,
        table_width=38,
    )

    it_preview = iter(loader)
    xm0, _, _ = next(it_preview)
    xm0 = [x.to(device) for x in xm0]
    print_model_layers_with_shapes(model, xm0)

    timer = _GPUEventTimer(print_every=log_every)

    step = 0
    prefetch = CudaPrefetcher(loader, device)
    model.train()

    with tqdm(total=steps, desc="sequence", dynamic_ncols=True, miniters=50) as pbar:
        while step < steps:
            with timer.cpu_range("data"):
                batch = prefetch.next()
                if batch is None:
                    prefetch = CudaPrefetcher(loader, device)
                    batch = prefetch.next()
                    if batch is None:
                        break
                xm, yp, m = batch

            timer.start_step()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                with timer.gpu_range("mov_enc"):
                    z_m = mov.encoder(xm)
                with timer.gpu_range("trunk"):
                    z_seq = model.trunk(z_m)
                b = z_seq.size(0)
                flat = z_seq.reshape(b * seq_len, latent_dim)
                with timer.gpu_range("ppl_dec"):
                    outs = per.decoder(flat)
                preds = [y.view(b, seq_len, *y.shape[1:]) for y in outs]
                with timer.gpu_range("rec"):
                    rec_loss, field_breakdown = _sequence_loss_and_breakdown(per.fields, preds, yp, m)
                with timer.gpu_range("tgt_enc"):
                    with torch.no_grad():
                        flat_targets = [y.view(b * seq_len, *y.shape[2:]) for y in yp]
                        z_tgt_flat = per.encoder(flat_targets)
                        z_tgt_seq = z_tgt_flat.view(b, seq_len, latent_dim)
                    nce_loss = _info_nce_masked_rows(z_seq, z_tgt_seq, m, temp)
                loss = w_lat * nce_loss + w_rec * rec_loss

            with timer.gpu_range("backward"):
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
            with timer.gpu_range("opt"):
                scaler.step(opt)
                scaler.update()

            step += 1
            pbar.update(1)

            vals = timer.end_step_and_accumulate()
            if vals is not None:
                keys, out = vals
                timer.print_line(keys, out, step)
                run_logger.add_scalars(
                    float(loss.detach().cpu().item()),
                    float(rec_loss.detach().cpu().item()),
                    float(nce_loss.detach().cpu().item()),
                    out["total"] / 1000.0,
                    opt,
                )
                run_logger.add_field_losses("loss/sequence_people", field_breakdown)
                run_logger.tick(
                    float(loss.detach().cpu().item()),
                    float(rec_loss.detach().cpu().item()),
                    float(nce_loss.detach().cpu().item()),
                )

            seq_logger.on_batch_end(step)

            if save_every and step % save_every == 0:
                out = Path(config["model_dir"])
                out.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), out / f"MovieToPeopleSequencePredictor_step_{step}.pt")

    out = Path(config["model_dir"])
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "MovieToPeopleSequencePredictor_final.pt")
    run_logger.close()


def main():
    parser = argparse.ArgumentParser(description="Train movie-to-people sequence decoder with latent supervision")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--movie-limit", type=int, default=0)
    args = parser.parse_args()
    ml = args.movie_limit if args.movie_limit > 0 else None
    train_sequence_predictor(project_config, steps=args.steps, save_every=args.save_every, movie_limit=ml)


if __name__ == "__main__":
    main()
