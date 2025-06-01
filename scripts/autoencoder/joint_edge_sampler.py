from __future__ import annotations
import sqlite3, numpy as np
from typing import List, Tuple, Dict
from prettytable import PrettyTable
from tqdm import tqdm


def _tensor_to_string(field, tensor):
    arr = np.array(tensor)
    if hasattr(field, "tokenizer") and field.tokenizer is not None:
        if arr.ndim >= 2 and arr.shape[-1] == field.tokenizer.get_vocab_size():
            arr = np.argmax(arr, axis=-1)
    if arr.ndim > 1:
        arr = arr.flatten()
    try:
        return field.to_string(arr)
    except Exception:
        return "[conv‑err]"


class AliasSampler:
    def __init__(self, probs: np.ndarray):
        n = len(probs)
        self.n = n
        self.p = np.zeros(n, dtype=np.float32)
        self.a = np.zeros(n, dtype=np.int32)

        scaled = probs * n
        small, large = [], []
        for i, v in enumerate(scaled):
            (small if v < 1.0 else large).append(i)

        while small and large:
            s, l = small.pop(), large.pop()
            self.p[s] = scaled[s]
            self.a[s] = l
            scaled[l] = (scaled[l] - 1) + scaled[s]
            (small if scaled[l] < 1.0 else large).append(l)

        for i in large + small:
            self.p[i] = 1.0
            self.a[i] = i

    def draw(self, k: int) -> np.ndarray:
        i = np.random.randint(0, self.n, size=k)
        accept = np.random.random(size=k) < self.p[i]
        return np.where(accept, i, self.a[i])


class WeightedEdgeSampler:
    """Edge sampler that builds tensors on demand and validates IO pairs."""
    def __init__(
        self,
        db_path: str,
        movie_ae,
        person_ae,
        batch_size: int,
        refresh_batches: int = 1_000,
        boost: float = 0.10,
        log_every: int = 512,          # <─ new
    ):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.mov = movie_ae
        self.per = person_ae
        self.bs = batch_size
        self.refresh_edges = max(1, refresh_batches) * batch_size
        self.boost = boost
        self.log_every = max(1, log_every)

        self.edges = self._load_edges()
        self.edge_to_idx = {eid: i for i, (eid, _, _) in enumerate(self.edges)}
        self.movie_tensors = [None] * len(self.edges)
        self.person_tensors = [None] * len(self.edges)

        self.mov_cache: Dict[str, Tuple] = {}
        self.per_cache: Dict[str, Tuple] = {}

        self.movie_cur = self.conn.cursor()
        self.person_cur = self.conn.cursor()
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

    # ----------------------------------------------------------- iterator
    def __iter__(self):
        return self

    def __next__(self):
        if self.seen % self.refresh_edges == 0:
            self._refresh_weights()

        idx = self.alias.draw(1)[0]
        self.seen += 1
        movie_t, person_t = self._get_tensors(idx)

        if self.seen % self.log_every == 1:              # ── log a sample
            eid, tconst, nconst = self.edges[idx]
            self._log_sample(eid, tconst, nconst, movie_t, person_t)

        return movie_t, person_t, self.edges[idx][0]

    # -------------------------------------------------------- tensor cache
    def _get_tensors(self, idx: int):
        if self.movie_tensors[idx] is None:
            eid, tconst, nconst = self.edges[idx]
            mr = self._movie_row(tconst)
            pr = self._person_row(nconst)
            self.movie_tensors[idx] = tuple(
                f.transform(mr.get(f.name)) for f in self.mov.fields
            )
            self.person_tensors[idx] = tuple(
                f.transform(pr.get(f.name)) for f in self.per.fields
            )
        return self.movie_tensors[idx], self.person_tensors[idx]

    # ----------------------------------------------------------- preload
    def _load_edges(self) -> List[Tuple[int, str, str]]:
        cur = self.conn.cursor()
        cur.execute("SELECT edgeId,tconst,nconst FROM edges;")
        return cur.fetchall()

    # ----------------------------------------------------------- SQL helpers
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

    # ------------------------------------------------------------ weights
    def _refresh_weights(self):
        cur = self.conn.cursor()
        try:
            cur.execute("SELECT edgeId,total_loss FROM edge_losses;")
            recorded = {eid: loss for eid, loss in cur.fetchall()}
            loss_vec = np.full(len(self.edges), 1000.0, dtype=np.float32)
            for eid, loss in recorded.items():
                idx = self.edge_to_idx.get(eid)
                if idx is not None:
                    loss_vec[idx] = loss
            lo, hi = loss_vec.min(), loss_vec.max()
            if hi > lo:
                norm = (loss_vec - lo) / (hi - lo)
                self.weights = 1.0 + self.boost * norm
            else:
                self.weights.fill(1.0)
        except sqlite3.Error:
            self.weights.fill(1.0)

        probs = self.weights / self.weights.sum()
        self.alias = AliasSampler(probs)

    # ------------------------------------------------------------ logging
    def _log_sample(self, eid, tconst, nconst, movie_t, person_t):
        mr = self._movie_row(tconst)
        pr = self._person_row(nconst)

        tm = PrettyTable(["movie", "orig", "tensor→str"])
        for f, t in zip(self.mov.fields, movie_t):
            tm.add_row([f.name,
                        str(mr.get(f.name))[:38],
                        _tensor_to_string(f, t)[:38]])

        tp = PrettyTable(["person", "orig", "tensor→str"])
        for f, t in zip(self.per.fields, person_t):
            tp.add_row([f.name,
                        str(pr.get(f.name))[:38],
                        _tensor_to_string(f, t)[:38]])

        print(f"\n=== sampler‑check eid={eid} ===")
        print(tm)
        print(tp)
        print("=" * 60)


def make_edge_sampler(
    db_path: str,
    movie_ae,
    person_ae,
    batch_size: int,
    refresh_batches: int = 1_000,
    boost: float = 0.10,
):
    return WeightedEdgeSampler(
        db_path,
        movie_ae,
        person_ae,
        batch_size=batch_size,
        refresh_batches=refresh_batches,
        boost=boost,
    )
