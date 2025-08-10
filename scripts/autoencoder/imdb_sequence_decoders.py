import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

from autoencoder.imdb_row_autoencoders import TitlesAutoencoder, PeopleAutoencoder
from autoencoder.sequence_reconstruction_logger import SequenceReconstructionLogger

class _SeqResidual(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.ln = nn.LayerNorm(width)
        self.c1 = nn.Conv1d(width, width, 1)
        self.c2 = nn.Conv1d(width, width, 1)
    def forward(self, x):
        r = x
        x = self.ln(x.transpose(1, 2)).transpose(1, 2)
        x = F.gelu(self.c1(x))
        x = self.c2(x)
        x = F.gelu(x + r)
        return x

class MoviesToPeopleSequenceDecoder(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.db_path = config["db_path"]
        self.latent_dim = int(config["latent_dim"])
        self.seq_len = int(config["people_sequence_length"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.movie_ae = TitlesAutoencoder(config)
        self.people_ae = PeopleAutoencoder(config)

        self.movie_ae.accumulate_stats()
        self.movie_ae.finalize_stats()
        self.movie_ae.build_autoencoder()
        self.movie_ae.load_model()

        self.people_ae.accumulate_stats()
        self.people_ae.finalize_stats()
        self.people_ae.build_autoencoder()
        self.people_ae.load_model()

        for p in self.movie_ae.encoder.parameters():
            p.requires_grad = False
        for p in self.people_ae.decoder.parameters():
            p.requires_grad = False

        width = self.latent_dim * 2
        self.project_up = nn.Conv1d(self.latent_dim, width, 1)
        self.blocks = nn.ModuleList([_SeqResidual(width) for _ in range(4)])
        self.project_down = nn.Conv1d(width, self.latent_dim, 1)

        wanted = config.get("sequence_active_people_fields")
        if wanted:
            names = [f.name for f in self.people_ae.fields]
            self.active_idx = [names.index(n) for n in wanted if n in names]
        else:
            self.active_idx = list(range(len(self.people_ae.fields)))

        self.to(self.device)
        self.train()

        self._conn: Optional[sqlite3.Connection] = None
        self._movie_cur = None
        self._people_cur = None

    def _ensure_conn(self):
        if self._conn is not None:
            return
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA temp_store=MEMORY;")
        self._conn.execute("PRAGMA cache_size=-200000;")
        self._conn.execute("PRAGMA mmap_size=268435456;")
        self._conn.execute("PRAGMA busy_timeout=5000;")
        self._movie_cur = self._conn.cursor()
        self._people_cur = self._conn.cursor()

    def forward(self, movie_inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        z = self.movie_ae.encoder([x for x in movie_inputs])
        z = z.unsqueeze(1).expand(-1, self.seq_len, -1)
        x = z.transpose(1, 2)
        x = F.gelu(self.project_up(x))
        for b in self.blocks:
            x = b(x)
        x = F.gelu(self.project_down(x))
        z_seq = x.transpose(1, 2)
        B, T, D = z_seq.shape
        flat = z_seq.reshape(B * T, D)
        outs = []
        for i in self.active_idx:
            o = self.people_ae.decoder.decs[i](flat)
            outs.append(_reshape_seq(o, B, T))
        return outs

    def compute_loss(self, preds: List[torch.Tensor], targets: List[torch.Tensor]) -> torch.Tensor:
        total = 0.0
        for idx, pred in zip(self.active_idx, preds):
            f = self.people_ae.fields[idx]
            tgt = targets[idx]
            B, T = pred.size(0), pred.size(1)
            p = _merge_bt(pred, B, T)
            y = _merge_bt(tgt, B, T)
            total = total + f.compute_loss(p, y) * float(f.weight)
        return total

    def _row_generator(self):
        self._ensure_conn()
        cur = self._movie_cur
        cur.execute(
            """
            SELECT
                t.tconst, t.primaryTitle, t.startYear, t.endYear,
                t.runtimeMinutes, t.averageRating, t.numVotes,
                (SELECT GROUP_CONCAT(genre, ',') FROM title_genres g WHERE g.tconst = t.tconst)
            FROM titles t
            WHERE
                t.startYear IS NOT NULL
                AND t.averageRating IS NOT NULL
                AND t.runtimeMinutes IS NOT NULL
                AND t.runtimeMinutes >= 5
                AND t.startYear >= 1850
                AND t.titleType IN ('movie','tvSeries','tvMovie','tvMiniSeries')
                AND t.numVotes >= 10
            """
        )
        ppl_sql = """
            SELECT
                p.primaryName, p.birthYear, p.deathYear,
                (SELECT GROUP_CONCAT(profession, ',') FROM people_professions pp WHERE pp.nconst = p.nconst)
            FROM people p
            INNER JOIN principals pr ON pr.nconst = p.nconst
            WHERE pr.tconst = ? AND p.birthYear IS NOT NULL
            GROUP BY p.nconst
            HAVING COUNT(1) > 0
            ORDER BY pr.ordering
            LIMIT ?
        """
        for tconst, primaryTitle, startYear, endYear, runtime, rating, votes, genres_str in cur:
            m_row = {
                "primaryTitle": primaryTitle,
                "startYear": startYear,
                "genres": genres_str.split(",") if genres_str else [],
            }
            self._people_cur.execute(ppl_sql, (tconst, self.seq_len))
            ppl = []
            for pn, by, dy, profs in self._people_cur.fetchall():
                p_row = {
                    "primaryName": pn,
                    "birthYear": by,
                }
                ppl.append(p_row)
            if not ppl:
                continue
            if len(ppl) < self.seq_len:
                ppl = ppl + [ppl[-1]] * (self.seq_len - len(ppl))
            else:
                ppl = ppl[: self.seq_len]
            yield m_row, ppl

    def make_loader(self) -> DataLoader:
        fields_m = self.movie_ae.fields
        fields_p = self.people_ae.fields
        bs = int(self.config.get("batch_size", 32))

        class _SeqDS(IterableDataset):
            def __iter__(ds_self):
                for m_row, ppl in self._row_generator():
                    X = [f.transform(m_row.get(f.name)) for f in fields_m]
                    Y = []
                    for i, f in enumerate(fields_p):
                        if i not in self.active_idx:
                            Y.append(None)
                            continue
                        seq = [f.transform_target(p.get(f.name)) for p in ppl]
                        Y.append(torch.stack(seq, dim=0))
                    yield X, Y

        def _collate(batch):
            mx = list(zip(*[b[0] for b in batch]))
            my = list(zip(*[b[1] for b in batch]))
            M = [torch.stack(col, dim=0) for col in mx]
            P = []
            for i, col in enumerate(my):
                if i not in self.active_idx:
                    continue
                P.append(torch.stack(col, dim=0))
            return M, P

        num_workers = int(self.config.get("num_workers", 0))
        prefetch = int(self.config.get("prefetch_factor", 2)) if num_workers > 0 else None
        pin = bool(torch.cuda.is_available())
        return DataLoader(
            _SeqDS(),
            batch_size=bs,
            collate_fn=_collate,
            num_workers=num_workers,
            prefetch_factor=prefetch,
            persistent_workers=True if num_workers > 0 else False,
            pin_memory=pin,
        )

    def fit(self):
        loader = self.make_loader()
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.config.get("learning_rate", 5e-4)),
            weight_decay=float(self.config.get("weight_decay", 1e-4)),
        )
        epochs = int(self.config.get("epochs", 10))
        scaler = GradScaler(enabled=(self.device.type == "cuda"))
        recon_logger = SequenceReconstructionLogger(
            seq_model=self,
            interval_steps=int(self.config.get("recon_log_interval", 200)),
            num_samples=int(self.config.get("row_recon_samples", 3)),
            timesteps_to_show=3,
            table_width=38,
            max_movie_scan=5000,
        )
        self.train()
        global_step = 0
        for epoch in range(epochs):
            pbar = tqdm(loader, desc=f"SeqDecoder epoch {epoch+1}/{epochs}")
            for M, P in pbar:
                M = [m.to(self.device, non_blocking=True) for m in M]
                P = [p.to(self.device, non_blocking=True) for p in P]
                with autocast(enabled=(self.device.type == "cuda")):
                    preds = self.forward(M)
                    loss = self.compute_loss(preds, P)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                pbar.set_postfix(loss=float(loss.detach().cpu().item()))
                recon_logger.on_batch_end(global_step)
                global_step += 1
        self.save_model()

    def save_model(self):
        out = Path(self.config["model_dir"])
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), out / "MoviesToPeopleSequenceDecoder.pt")

    def load_model(self):
        p = Path(self.config["model_dir"]) / "MoviesToPeopleSequenceDecoder.pt"
        if p.exists():
            self.load_state_dict(torch.load(p, map_location=self.device))
            self.eval()
        return self

def _reshape_seq(t: torch.Tensor, B: int, T: int) -> torch.Tensor:
    if t.dim() == 2:
        return t.view(B, T, -1)
    if t.dim() == 3:
        C = t.size(1)
        return t.view(B, T, C, -1)
    if t.dim() == 4:
        return t.view(B, T, t.size(1), t.size(2), t.size(3))
    return t

def _merge_bt(x: torch.Tensor, B: int, T: int) -> torch.Tensor:
    s = list(x.shape)
    s[0] = B * T
    s.pop(1)
    return x.contiguous().view(*s)
