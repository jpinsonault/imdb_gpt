# scripts/autoencoder/mapping_samplers.py
from __future__ import annotations
import sqlite3
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
from torch.utils.data import IterableDataset

from .fields import BaseField


class LossLedger:
    _BULK_SIZE = 10000

    def __init__(self, default_loss: float = 1000.0):
        self._cache: List[Tuple[int, float]] = []
        self._loss_map: Dict[int, float] = {}
        self._default = float(default_loss)

    def add(self, key_id: int, total_loss: float):
        self._cache.append((int(key_id), float(total_loss)))
        if len(self._cache) >= self._BULK_SIZE:
            self.flush()

    def flush(self):
        if not self._cache:
            return
        for k, v in self._cache:
            self._loss_map[k] = v
        self._cache.clear()

    def snapshot(self) -> Dict[int, float]:
        self.flush()
        return dict(self._loss_map)

    def close(self):
        self.flush()
        self._loss_map.clear()

    @property
    def default_loss(self) -> float:
        return self._default


class AliasSampler:
    def __init__(self, probs: np.ndarray):
        n = int(len(probs))
        self.n = n
        self.p = np.zeros(n, dtype=np.float32)
        self.a = np.zeros(n, dtype=np.int32)
        scaled = np.asarray(probs, dtype=np.float64) * n
        small, large = [], []
        for i, v in enumerate(scaled):
            (small if v < 1.0 else large).append(i)
        while small and large:
            s, l = small.pop(), large.pop()
            self.p[s] = scaled[s]
            self.a[s] = l
            scaled[l] = (scaled[l] - 1.0) + scaled[s]
            (small if scaled[l] < 1.0 else large).append(l)
        for i in large + small:
            self.p[i] = 1.0
            self.a[i] = i

    def draw(self, k: int) -> np.ndarray:
        i = np.random.randint(0, self.n, size=int(k))
        accept = np.random.random(size=int(k)) < self.p[i]
        return np.where(accept, i, self.a[i])


class MoviePeoplePairSampler:
    def __init__(
        self,
        db_path: str,
        movie_fields: List[BaseField],
        people_fields: List[BaseField],
        seq_len: int,
        batch_size: int,
        refresh_batches: int = 1000,
        boost: float = 0.10,
        loss_logger: Optional[LossLedger] = None,
    ):
        self.db_path = db_path
        self.movie_fields = movie_fields
        self.people_fields = people_fields
        self.seq_len = int(seq_len)
        self.bs = int(batch_size)
        self.refresh_edges = max(1, int(refresh_batches)) * self.bs
        self.boost = float(boost)
        self.loss_logger = loss_logger

        self.keys: List[Tuple[str, int, str]] = self._load_keys()
        self.key_to_idx: Dict[Tuple[str, int, str], int] = {k: i for i, k in enumerate(self.keys)}

        self.mov_cache_row: Dict[str, Dict] = {}
        self.per_cache_row: Dict[str, Dict] = {}
        self.mov_cache_tensor: Dict[str, Tuple[torch.Tensor, ...]] = {}

        self.conn = None
        self.cur_movie = None
        self.cur_person = None

        self.movie_sql = """
        SELECT primaryTitle,startYear,endYear,runtimeMinutes,
               averageRating,numVotes,
               (SELECT GROUP_CONCAT(genre,',') FROM title_genres WHERE tconst = ?)
        FROM titles WHERE tconst = ? LIMIT 1
        """
        self.person_sql = """
        SELECT primaryName,birthYear,deathYear,
               (SELECT GROUP_CONCAT(profession,',') FROM people_professions WHERE nconst = ?)
        FROM people WHERE nconst = ? LIMIT 1
        """

        self.weights = np.ones(len(self.keys), dtype=np.float32)
        self.alias = AliasSampler(self.weights / self.weights.sum())
        self.seen = 0

    def _ensure_conn(self):
        if self.conn is not None:
            return
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA synchronous = NORMAL;")
        self.conn.execute("PRAGMA temp_store = MEMORY;")
        self.conn.execute("PRAGMA cache_size = -200000;")
        self.conn.execute("PRAGMA mmap_size = 268435456;")
        self.conn.execute("PRAGMA busy_timeout = 5000;")
        self.cur_movie = self.conn.cursor()
        self.cur_person = self.conn.cursor()

    def __iter__(self):
        self._ensure_conn()
        return self

    def __next__(self):
        if self.seen % self.refresh_edges == 0:
            self._refresh_weights()
        idx = int(self.alias.draw(1)[0])
        self.seen += 1
        xm, yp, m = self._get_tensors(idx)
        key_id = idx
        return xm, yp, m, key_id

    def _load_keys(self) -> List[Tuple[str, int, str]]:
        sql = f"""
        SELECT pr.tconst, pr.ordering, pr.nconst
        FROM principals pr
        JOIN titles t ON t.tconst = pr.tconst
        JOIN title_genres g ON g.tconst = t.tconst
        JOIN people p ON p.nconst = pr.nconst
        LEFT JOIN people_professions pp ON pp.nconst = p.nconst
        WHERE
            t.startYear IS NOT NULL
            AND t.startYear >= 1850
            AND t.averageRating IS NOT NULL
            AND t.runtimeMinutes IS NOT NULL
            AND t.runtimeMinutes >= 5
            AND t.titleType IN ('movie','tvSeries','tvMovie','tvMiniSeries')
            AND t.numVotes >= 10
            AND p.birthYear IS NOT NULL
            AND pr.ordering BETWEEN 1 AND ?
        GROUP BY pr.tconst, pr.ordering, pr.nconst
        HAVING COUNT(pp.profession) > 0
        """
        out: List[Tuple[str, int, str]] = []
        with sqlite3.connect(self.db_path, check_same_thread=False) as con:
            for tconst, ordering, nconst in con.execute(sql, (self.seq_len,)):
                out.append((str(tconst), int(ordering), str(nconst)))
        return out

    def _movie_row(self, tconst: str) -> Dict:
        if tconst in self.mov_cache_row:
            return self.mov_cache_row[tconst]
        r = self.cur_movie.execute(self.movie_sql, (tconst, tconst)).fetchone()
        row = {
            "tconst": tconst,
            "primaryTitle": r[0],
            "startYear": r[1],
            "endYear": r[2],
            "runtimeMinutes": r[3],
            "averageRating": r[4],
            "numVotes": r[5],
            "genres": r[6].split(",") if r[6] else [],
        }
        self.mov_cache_row[tconst] = row
        return row

    def _person_row(self, nconst: str) -> Dict:
        if nconst in self.per_cache_row:
            return self.per_cache_row[nconst]
        r = self.cur_person.execute(self.person_sql, (nconst, nconst)).fetchone()
        row = {
            "primaryName": r[0],
            "birthYear": r[1],
            "deathYear": r[2],
            "professions": r[3].split(",") if r[3] else None,
        }
        self.per_cache_row[nconst] = row
        return row

    def _movie_tensors(self, tconst: str) -> Tuple[torch.Tensor, ...]:
        if tconst in self.mov_cache_tensor:
            return self.mov_cache_tensor[tconst]
        mr = self._movie_row(tconst)
        xs = tuple(f.transform(mr.get(f.name)) for f in self.movie_fields)
        self.mov_cache_tensor[tconst] = xs
        return xs

    def _get_tensors(self, idx: int):
        tconst, ordering, nconst = self.keys[idx]
        xm = self._movie_tensors(tconst)
        pr = self._person_row(nconst)

        yp: List[torch.Tensor] = []
        pos = max(0, min(self.seq_len, int(ordering))) - 1
        for f in self.people_fields:
            steps = []
            for t in range(self.seq_len):
                if t == pos:
                    steps.append(f.transform_target(pr.get(f.name)))
                else:
                    steps.append(f.get_base_padding_value())
            yp.append(torch.stack(steps, dim=0))

        m = torch.zeros(self.seq_len, dtype=torch.float32)
        m[pos] = 1.0
        return xm, yp, m

    def _refresh_weights(self):
        if self.loss_logger is None:
            self.weights.fill(1.0)
            self.alias = AliasSampler(self.weights / self.weights.sum())
            return

        recorded = self.loss_logger.snapshot()
        loss_vec = np.full(len(self.keys), float(self.loss_logger.default_loss), dtype=np.float32)
        for k_idx in range(len(self.keys)):
            if k_idx in recorded:
                loss_vec[k_idx] = float(recorded[k_idx])

        lo = float(loss_vec.min())
        hi = float(loss_vec.max())
        if hi > lo:
            norm = (loss_vec - lo) / (hi - lo)
            self.weights = 1.0 + self.boost * norm.astype(np.float32)
        else:
            self.weights.fill(1.0)

        probs = self.weights / self.weights.sum()
        self.alias = AliasSampler(probs)


class SequencePairIterable(IterableDataset):
    def __init__(self, sampler: MoviePeoplePairSampler):
        super().__init__()
        self.sampler = sampler

    def __iter__(self):
        for xm, yp, m, key_id in self.sampler:
            yield xm, yp, m, key_id


def collate_pairs(batch):
    xm, yp, m, k = zip(*batch)
    xm_cols = list(zip(*xm))
    yp_cols = list(zip(*yp))
    Xm = [torch.stack(col, dim=0) for col in xm_cols]
    Yp = [torch.stack(col, dim=0) for col in yp_cols]
    M = torch.stack(m, dim=0)
    K = torch.tensor(k, dtype=torch.long)
    return Xm, Yp, M, K
