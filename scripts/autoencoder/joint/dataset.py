from __future__ import annotations
import sqlite3
import time
import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from multiprocessing import shared_memory

class _Alias:
    @staticmethod
    def build_from_probs(probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = int(probs.size)
        p = np.zeros(n, dtype=np.float32)
        a = np.zeros(n, dtype=np.int32)
        scaled = probs * n
        small, large = [], []
        for i, v in enumerate(scaled):
            (small if v < 1.0 else large).append(i)
        while small and large:
            s = small.pop()
            l = large.pop()
            p[s] = scaled[s]
            a[s] = l
            scaled[l] = (scaled[l] - 1.0) + scaled[s]
            (small if scaled[l] < 1.0 else large).append(l)
        for i in large + small:
            p[i] = 1.0
            a[i] = i
        return p, a

class SharedSamplerState:
    def __init__(self, n: int, init_probs: Optional[np.ndarray] = None):
        self.n = int(n)
        self._p_shm = shared_memory.SharedMemory(create=True, size=self.n * 4)
        self._a_shm = shared_memory.SharedMemory(create=True, size=self.n * 4)
        self._p = np.ndarray((self.n,), dtype=np.float32, buffer=self._p_shm.buf)
        self._a = np.ndarray((self.n,), dtype=np.int32, buffer=self._a_shm.buf)
        if init_probs is None:
            init_probs = np.ones(self.n, dtype=np.float32) / max(1, self.n)
        p, a = _Alias.build_from_probs(init_probs.astype(np.float32, copy=False))
        self._p[:] = p
        self._a[:] = a

    def __getstate__(self):
        return {
            "n": self.n,
            "p_name": self._p_shm.name,
            "a_name": self._a_shm.name,
        }

    def __setstate__(self, state):
        self.n = int(state["n"])
        self._p_shm = shared_memory.SharedMemory(name=state["p_name"])
        self._a_shm = shared_memory.SharedMemory(name=state["a_name"])
        self._p = np.ndarray((self.n,), dtype=np.float32, buffer=self._p_shm.buf)
        self._a = np.ndarray((self.n,), dtype=np.int32, buffer=self._a_shm.buf)

    def arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._p, self._a

    def update_alias(self, probs: np.ndarray):
        probs = probs.astype(np.float32, copy=False)
        probs_sum = float(probs.sum())
        if probs_sum <= 0.0:
            probs = np.ones_like(probs, dtype=np.float32) / max(1, probs.size)
        else:
            probs = probs / probs_sum
        p, a = _Alias.build_from_probs(probs)
        self._p[:] = p
        self._a[:] = a

class WeightedEdgeSampler:
    def __init__(
        self,
        db_path: str,
        movie_ae,
        person_ae,
        batch_size: int,
        boost: float = 0.10,
        shared_state: Optional[SharedSamplerState] = None,
    ):
        self.db_path = db_path
        self.mov = movie_ae
        self.per = person_ae
        self.bs = batch_size
        self.boost = boost

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

        if shared_state is None:
            init_probs = np.ones(len(self.edges), dtype=np.float32) / max(1, len(self.edges))
            self.state = SharedSamplerState(len(self.edges), init_probs)
        else:
            self.state = shared_state

    def _ensure_conn(self):
        if self.conn is not None:
            return
        t0 = time.perf_counter()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA synchronous = NORMAL;")
        self.conn.execute("PRAGMA temp_store = MEMORY;")
        self.conn.execute("PRAGMA cache_size = -200000;")
        self.conn.execute("PRAGMA mmap_size = 268435456;")
        self.conn.execute("PRAGMA busy_timeout = 5000;")
        self.movie_cur = self.conn.cursor()
        self.person_cur = self.conn.cursor()
        logging.info(f"edge sampler: DB connection ready in {time.perf_counter() - t0:.2f}s")

    def __iter__(self):
        self._ensure_conn()
        return self

    def __next__(self):
        p, a = self.state.arrays()
        n = int(p.size)
        if n == 0:
            raise StopIteration
        i = np.random.randint(0, n, size=1)[0]
        accept = float(np.random.random(1)[0]) < float(p[i])
        idx = int(i) if accept else int(a[i])
        movie_t, person_t = self._get_tensors(idx)
        return movie_t, person_t, self.edges[idx][0]

    def _get_tensors(self, idx: int):
        if self.movie_tensors[idx] is None:
            eid, tconst, nconst = self.edges[idx]
            mr = self._movie_row(tconst)
            pr = self._person_row(nconst)
            self.movie_tensors[idx] = tuple(f.transform(mr.get(f.name)) for f in self.mov.fields)
            self.person_tensors[idx] = tuple(f.transform(pr.get(f.name)) for f in self.per.fields)
        return self.movie_tensors[idx], self.person_tensors[idx]

    def _load_edges(self) -> List[Tuple[int, str, str]]:
        t0 = time.perf_counter()
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            cur = conn.cursor()
            total = None
            try:
                total = cur.execute("SELECT COUNT(1) FROM edges;").fetchone()[0]
            except sqlite3.Error:
                pass
            if total is not None:
                logging.info(f"edge sampler: loading edges table ({total} rows)…")
            else:
                logging.info("edge sampler: loading edges table…")
            cur.execute("SELECT edgeId,tconst,nconst FROM edges;")
            rows = cur.fetchall()
        logging.info(f"edge sampler: edges loaded ({len(rows)} rows) in {time.perf_counter() - t0:.2f}s")
        return rows

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

    def _bulk_fetch_movies(self, tconsts: List[str]):
        self._ensure_conn()
        missing = [t for t in tconsts if t not in self.mov_cache]
        if not missing:
            return
        chunk = 512
        for i in range(0, len(missing), chunk):
            part = missing[i:i+chunk]
            q = (
                "SELECT t.tconst, t.primaryTitle, t.startYear, t.endYear, "
                "t.runtimeMinutes, t.averageRating, t.numVotes, "
                "(SELECT GROUP_CONCAT(genre, ',') FROM title_genres WHERE tconst = t.tconst) "
                "FROM titles t WHERE t.tconst IN (" + ",".join(["?"] * len(part)) + ");"
            )
            for r in self.movie_cur.execute(q, tuple(part)).fetchall():
                tconst = r[0]
                self.mov_cache[tconst] = {
                    "tconst": tconst,
                    "primaryTitle": r[1],
                    "startYear": r[2],
                    "endYear": r[3],
                    "runtimeMinutes": r[4],
                    "averageRating": r[5],
                    "numVotes": r[6],
                    "genres": r[7].split(",") if r[7] else [],
                }

    def _bulk_fetch_people(self, nconsts: List[str]):
        self._ensure_conn()
        missing = [n for n in nconsts if n not in self.per_cache]
        if not missing:
            return
        chunk = 512
        for i in range(0, len(missing), chunk):
            part = missing[i:i+chunk]
            q = (
                "SELECT p.nconst, p.primaryName, p.birthYear, p.deathYear, "
                "(SELECT GROUP_CONCAT(profession, ',') FROM people_professions WHERE nconst = p.nconst) "
                "FROM people p WHERE p.nconst IN (" + ",".join(["?"] * len(part)) + ");"
            )
            for r in self.person_cur.execute(q, tuple(part)).fetchall():
                nconst = r[0]
                self.per_cache[nconst] = {
                    "primaryName": r[1],
                    "birthYear": r[2],
                    "deathYear": r[3],
                    "professions": r[4].split(",") if r[4] else None,
                }

    def sample_batch(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        self._ensure_conn()
        p, a = self.state.arrays()
        n = int(p.size)
        if n == 0:
            M_empty = [torch.empty((0,) + tuple(f.input_shape), dtype=torch.float32) for f in self.mov.fields]
            P_empty = [torch.empty((0,) + tuple(f.input_shape), dtype=torch.float32) for f in self.per.fields]
            e_empty = torch.empty((0,), dtype=torch.long)
            return M_empty, P_empty, e_empty

        bs = int(self.bs)
        i = np.random.randint(0, n, size=bs)
        accept = np.random.random(size=bs) < p[i]
        idx = np.where(accept, i, a[i]).astype(np.int64)

        eids = np.fromiter((self.edges[j][0] for j in idx), dtype=np.int64, count=bs)
        tconsts = [self.edges[j][1] for j in idx]
        nconsts = [self.edges[j][2] for j in idx]

        self._bulk_fetch_movies(tconsts)
        self._bulk_fetch_people(nconsts)

        m_cols: List[List[torch.Tensor]] = [[] for _ in self.mov.fields]
        p_cols: List[List[torch.Tensor]] = [[] for _ in self.per.fields]

        for tconst, nconst in zip(tconsts, nconsts):
            mr = self.mov_cache[tconst]
            pr = self.per_cache[nconst]
            for k, f in enumerate(self.mov.fields):
                m_cols[k].append(f.transform(mr.get(f.name)))
            for k, f in enumerate(self.per.fields):
                p_cols[k].append(f.transform(pr.get(f.name)))

        M = [torch.stack(col, dim=0) for col in m_cols]
        P = [torch.stack(col, dim=0) for col in p_cols]
        e = torch.tensor(eids, dtype=torch.long)
        return M, P, e

def make_edge_sampler(
    db_path: str,
    movie_ae,
    person_ae,
    batch_size: int,
    boost: float = 0.10,
    shared_state: Optional[SharedSamplerState] = None,
):
    return WeightedEdgeSampler(
        db_path,
        movie_ae,
        person_ae,
        batch_size=batch_size,
        boost=boost,
        shared_state=shared_state,
    )
