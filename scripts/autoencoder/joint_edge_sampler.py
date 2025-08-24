from __future__ import annotations
import logging
import sqlite3
import numpy as np
from typing import List, Tuple, Dict, Optional

from .mapping_samplers import AliasSampler


class WeightedEdgeSampler:
    def __init__(
        self,
        db_path: str,
        movie_ae,
        person_ae,
        batch_size: int,
        refresh_batches: int = 1_000,
        boost: float = 0.10,
        loss_logger=None,
    ):
        self.db_path = db_path
        self.mov = movie_ae
        self.per = person_ae
        self.bs = batch_size
        self.refresh_edges = max(1, refresh_batches) * batch_size
        self.boost = boost
        self.loss_logger = loss_logger

        self.edges = self._load_edges()
        self.edge_to_idx = {eid: i for i, (eid, _, _) in enumerate(self.edges)}
        self.movie_tensors = [None] * len(self.edges)
        self.person_tensors = [None] * len(self.edges)

        self.mov_cache: Dict[str, Dict] = {}
        self.per_cache: Dict[str, Dict] = {}

        self.conn = None
        self.movie_cur = None
        self.person_cur = None

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

        self.weights = np.ones(len(self.edges), dtype=np.float32)
        self.alias = AliasSampler(self.weights / self.weights.sum())
        self.seen = 0
        self._cache_misses = 0

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
        self.movie_cur = self.conn.cursor()
        self.person_cur = self.conn.cursor()

    def __iter__(self):
        self._ensure_conn()
        return self

    def __next__(self):
        if self.seen % self.refresh_edges == 0:
            self._refresh_weights()
        idx = self.alias.draw(1)[0]
        self.seen += 1
        movie_t, person_t = self._get_tensors(idx)
        return movie_t, person_t, self.edges[idx][0]

    def _get_tensors(self, idx: int):
        if self.movie_tensors[idx] is None:
            self._cache_misses += 1
            eid, tconst, nconst = self.edges[idx]
            mr = self._movie_row(tconst)
            pr = self._person_row(nconst)
            self.movie_tensors[idx] = tuple(f.transform(mr.get(f.name)) for f in self.mov.fields)
            self.person_tensors[idx] = tuple(f.transform(pr.get(f.name)) for f in self.per.fields)
        return self.movie_tensors[idx], self.person_tensors[idx]

    def _load_edges(self) -> List[Tuple[int, str, str]]:
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            cur = conn.cursor()
            cur.execute("SELECT edgeId,tconst,nconst FROM edges;")
            return cur.fetchall()

    def _movie_row(self, tconst: str):
        if tconst in self.mov_cache:
            return self.mov_cache[tconst]
        r = self.movie_cur.execute(self.movie_sql, (tconst, tconst)).fetchone()
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
        self.mov_cache[tconst] = row
        return row

    def _person_row(self, nconst: str):
        if nconst in self.per_cache:
            return self.per_cache[nconst]
        r = self.person_cur.execute(self.person_sql, (nconst, nconst)).fetchone()
        row = {
            "primaryName": r[0],
            "birthYear": r[1],
            "deathYear": r[2],
            "professions": r[3].split(",") if r[3] else None,
        }
        self.per_cache[nconst] = row
        return row

    def _refresh_weights(self):
        if self.loss_logger is None:
            self.weights.fill(1.0)
            self.alias = AliasSampler(self.weights / self.weights.sum())
            return

        recorded = self.loss_logger.snapshot()
        default_loss = float(getattr(self.loss_logger, "default_loss", 1000.0))
        loss_vec = np.full(len(self.edges), default_loss, dtype=np.float32)
        for eid, loss in recorded.items():
            idx = self.edge_to_idx.get(eid)
            if idx is not None:
                loss_vec[idx] = float(loss)

        lo, hi = float(loss_vec.min()), float(loss_vec.max())
        if hi > lo:
            norm = (loss_vec - lo) / (hi - lo)
            self.weights = 1.0 + self.boost * norm.astype(np.float32)
        else:
            self.weights.fill(1.0)

        probs = self.weights / self.weights.sum()
        self.alias = AliasSampler(probs)


def make_edge_sampler(
    db_path: str,
    movie_ae,
    person_ae,
    batch_size: int,
    refresh_batches: int = 1_000,
    boost: float = 0.10,
    loss_logger=None,
):
    return WeightedEdgeSampler(
        db_path,
        movie_ae,
        person_ae,
        batch_size=batch_size,
        refresh_batches=refresh_batches,
        boost=boost,
        loss_logger=loss_logger,
    )
