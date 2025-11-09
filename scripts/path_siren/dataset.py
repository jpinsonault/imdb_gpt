import os
import sqlite3
from collections import OrderedDict
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Dict, Any

import torch
from torch.utils.data import IterableDataset

from config import project_config


class _LRU:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.data = OrderedDict()

    def get(self, key):
        if key not in self.data:
            return None
        v = self.data.pop(key)
        self.data[key] = v
        return v

    def put(self, key, value):
        if self.capacity <= 0:
            return
        if key in self.data:
            self.data.pop(key)
        elif len(self.data) >= self.capacity:
            self.data.popitem(last=False)
        self.data[key] = value


class TitlePathIterable(IterableDataset):
    """
    Each yielded sample is a dict with keys:
        Mx: list[Tensor]           movie inputs per field
        My: list[Tensor]           movie targets per field
        movie_latent: Tensor       (D,) movie latent
        Z_lat_tgts: Tensor         (L,D) latent trajectory (title + people)
        Yp_tgts: list[Tensor]      per-person-field target sequences, each (L,...)
        t_grid: Tensor             (L,) time positions
        time_mask: Tensor          (L,) 1 for valid positions
    """

    def __init__(
        self,
        db_path: str,
        principals_table: str,
        movie_ae,
        people_ae,
        num_people: int,
        shuffle: bool = True,
        cache_capacity_movies: int = 200000,
        cache_capacity_people: int = 400000,
        movie_limit: Optional[int] = None,
        seed: int = 1337,
        cache_file: Optional[str] = None,
        use_precomputed_cache: bool = True,
        # backwards-compat alias used by some callers
        path_siren_cache_path: Optional[str] = None,
    ):
        super().__init__()

        self.db_path = db_path
        self.principals_table = principals_table
        self.movie_ae = movie_ae
        self.people_ae = people_ae
        self.num_people = int(num_people)
        self.shuffle = bool(shuffle)
        self.movie_limit = None if movie_limit in (None, 0) else int(movie_limit)
        self.seed = int(seed)

        # ---------- precomputed cache wiring ----------
        if cache_file is None and path_siren_cache_path is not None:
            cache_file = path_siren_cache_path

        if cache_file is None:
            # default to the shared precompute target
            cache_file = str(Path(project_config.data_dir) / "path_siren_cache.pt")

        self.cache_file = cache_file
        self._precomputed: Optional[List[Any]] = None

        if use_precomputed_cache and os.path.exists(self.cache_file):
            print(f"[path-siren] using precomputed cache: {self.cache_file}")
            obj = torch.load(self.cache_file, map_location="cpu")

            # support {"samples": [...]} or raw list/tuple
            if isinstance(obj, dict) and "samples" in obj:
                self._precomputed = list(obj["samples"])
            elif isinstance(obj, (list, tuple)):
                self._precomputed = list(obj)
            else:
                raise ValueError(
                    f"[path-siren] Unexpected cache format in {self.cache_file}: "
                    f"expected dict with 'samples' or list, got {type(obj)}"
                )

        # ---------- online mode state (only used if no precomputed cache) ----------
        self._conn: Optional[sqlite3.Connection] = None
        self._cur: Optional[sqlite3.Cursor] = None

        self._movie_cache = _LRU(cache_capacity_movies)
        self._people_cache = _LRU(cache_capacity_people)
        self._z_movie_cache = _LRU(cache_capacity_movies)
        self._z_person_cache = _LRU(cache_capacity_people)

        self._movie_sql = """
        SELECT primaryTitle,startYear,endYear,runtimeMinutes,
               averageRating,numVotes,
               (SELECT GROUP_CONCAT(genre,',') FROM title_genres WHERE tconst = ?)
        FROM titles WHERE tconst = ? LIMIT 1
        """
        self._people_sql = """
        SELECT p.primaryName,birthYear,deathYear,
               (SELECT GROUP_CONCAT(profession,',') FROM people_professions WHERE nconst = p.nconst)
        FROM people p
        WHERE p.nconst = ? LIMIT 1
        """

    # ---------- precomputed fast path ----------

    def __iter__(self):
        # If we have a precomputed cache, yield from it and never touch SQLite.
        if self._precomputed is not None:
            n = len(self._precomputed)
            if n == 0:
                return iter(())

            idxs = list(range(n))
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.seed)
                perm = torch.randperm(n, generator=g).tolist()
                idxs = [idxs[i] for i in perm]

            for i in idxs:
                s = self._precomputed[i]

                # Normalization: accept both dict samples and legacy 7-tuples
                if isinstance(s, dict):
                    sample = s
                else:
                    # Expect: (Mx, My, Zt, Z_lat_tgts, Yp_tgts, t_grid, time_mask)
                    if not isinstance(s, (list, tuple)) or len(s) != 7:
                        raise ValueError(
                            f"[path-siren] Bad sample format in cache: index {i}, type={type(s)}, len={getattr(s, 'len', None)}"
                        )
                    Mx, My, Zt, Z_lat_tgts, Yp_tgts, t_grid, time_mask = s
                    sample = {
                        "Mx": Mx,
                        "My": My,
                        "movie_latent": Zt,
                        "Z_lat_tgts": Z_lat_tgts,
                        "Yp_tgts": Yp_tgts,
                        "t_grid": t_grid,
                        "time_mask": time_mask,
                    }

                # Ensure movie_latent key exists if older cache used "Zt"
                if "movie_latent" not in sample and "Zt" in sample:
                    sample["movie_latent"] = sample["Zt"]

                yield sample

            return

        # Otherwise fall back to original DB-based iterator.
        self._open()
        try:
            for tconst in self._iter_tconsts():
                yield self._build_sample_for_tconst(tconst)
        finally:
            self._close()

    # ---------- SQLite / online mode helpers ----------

    def _open(self):
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._cur = self._conn.cursor()

    def _close(self):
        if self._cur is not None:
            self._cur.close()
            self._cur = None
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _iter_tconsts(self) -> Iterator[str]:
        """
        Mirror the selection logic used in the precompute script.
        """
        seed = self.seed
        if self.movie_limit is not None:
            q = """
            SELECT DISTINCT tconst, (movie_hash + ?) AS k
            FROM edges
            ORDER BY k
            LIMIT ?
            """
            rows = self._cur.execute(q, (seed, self.movie_limit)).fetchall()
            tconsts = [r[0] for r in rows]
        else:
            rows = self._cur.execute("SELECT DISTINCT tconst FROM edges").fetchall()
            tconsts = [r[0] for r in rows]

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(seed)
            perm = torch.randperm(len(tconsts), generator=g).tolist()
            for i in perm:
                yield tconsts[i]
        else:
            for t in tconsts:
                yield t

    def _get_movie_row(self, tconst: str) -> Dict[str, Any]:
        cached = self._movie_cache.get(tconst)
        if cached is not None:
            return cached
        r = self._cur.execute(self._movie_sql, (tconst, tconst)).fetchone()
        if r is None:
            row = {
                "tconst": tconst,
                "primaryTitle": None,
                "startYear": None,
                "endYear": None,
                "runtimeMinutes": None,
                "averageRating": None,
                "numVotes": None,
                "genres": [],
            }
        else:
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
        self._movie_cache.put(tconst, row)
        return row

    def _get_person_row(self, nconst: str) -> Dict[str, Any]:
        cached = self._people_cache.get(nconst)
        if cached is not None:
            return cached
        r = self._cur.execute(self._people_sql, (nconst,)).fetchone()
        if r is None:
            row = {
                "primaryName": None,
                "birthYear": None,
                "deathYear": None,
                "professions": None,
            }
        else:
            row = {
                "primaryName": r[0],
                "birthYear": r[1],
                "deathYear": r[2],
                "professions": r[3].split(",") if r[3] else None,
            }
        self._people_cache.put(nconst, row)
        return row

    def _people_for_title(self, tconst: str) -> List[str]:
        rows = self._cur.execute(
            f"""
            SELECT pr.nconst
            FROM {self.principals_table} pr
            JOIN people p ON p.nconst = pr.nconst
            WHERE pr.tconst = ?
              AND p.birthYear IS NOT NULL
            ORDER BY pr.ordering
            LIMIT ?
            """,
            (tconst, self.num_people),
        ).fetchall()
        return [r[0] for r in rows]

    def _encode_movie(
        self,
        tconst: str,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        cached = self._z_movie_cache.get(tconst)
        if cached is not None:
            return cached

        row = self._get_movie_row(tconst)
        Mx: List[torch.Tensor] = []
        My: List[torch.Tensor] = []
        for f in self.movie_ae.fields:
            Mx.append(f.transform(row.get(f.name)))
            My.append(f.transform_target(row.get(f.name)))

        xs = [x.unsqueeze(0) for x in Mx]
        with torch.no_grad():
            z = self.movie_ae.encoder(xs).cpu().squeeze(0)
        self._z_movie_cache.put(tconst, (Mx, My, z))
        return Mx, My, z

    def _encode_person(self, nconst: str) -> Tuple[List[torch.Tensor], torch.Tensor]:
        cached = self._z_person_cache.get(nconst)
        if cached is not None:
            return cached

        row = self._get_person_row(nconst)
        X: List[torch.Tensor] = []
        for f in self.people_ae.fields:
            X.append(f.transform(row.get(f.name)))
        xs = [x.unsqueeze(0) for x in X]
        with torch.no_grad():
            z = self.people_ae.encoder(xs).cpu().squeeze(0)
        self._z_person_cache.put(nconst, (X, z))
        return X, z

    def _build_sample_for_tconst(self, tconst: str) -> Dict[str, Any]:
        # movie enc
        Mx, My, z_title = self._encode_movie(tconst)

        # people encs
        nconsts = self._people_for_title(tconst)
        L = self.num_people + 1  # title + up to N people
        t_grid = torch.linspace(0.0, 1.0, steps=L)

        per_field_targets: List[List[torch.Tensor]] = [
            [] for _ in self.people_ae.fields
        ]
        z_steps = [z_title]

        for n in nconsts:
            _, z_p = self._encode_person(n)
            prow = self._get_person_row(n)
            ys = []
            for f in self.people_ae.fields:
                ys.append(f.transform_target(prow.get(f.name)))
            for fi, y in enumerate(ys):
                per_field_targets[fi].append(y)
            z_steps.append(z_p)

        people_k = len(nconsts)
        valid_len = min(people_k + 1, L)
        time_mask = torch.zeros(L, dtype=torch.float32)
        time_mask[:valid_len] = 1.0

        # prepend dummy + pad for each people field
        Yp_tgts: List[torch.Tensor] = []
        for fi, f in enumerate(self.people_ae.fields):
            dummy0 = f.get_base_padding_value()
            seq = [dummy0] + per_field_targets[fi]
            if len(seq) < L:
                pad_needed = L - len(seq)
                seq += [dummy0] * pad_needed
            Yp_tgts.append(torch.stack(seq, dim=0))

        # pad z_steps by repeating last
        if len(z_steps) < L:
            last = z_steps[-1]
            z_steps += [last] * (L - len(z_steps))
        Z_lat_tgts = torch.stack(z_steps, dim=0)

        sample: Dict[str, Any] = {
            "Mx": Mx,
            "My": My,
            "movie_latent": z_title,
            "Z_lat_tgts": Z_lat_tgts,
            "Yp_tgts": Yp_tgts,
            "t_grid": t_grid,
            "time_mask": time_mask,
        }
        return sample


# ---------- collate function ----------

def collate_batch(batch: List[Dict[str, Any]]):
    """
    Batch a list of samples produced by TitlePathIterable.

    Returns:
        Mx_batched: list[Tensor]          movie inputs per field, each (B, ...)
        My_batched: list[Tensor]          movie targets per field, each (B, ...)
        Zt: Tensor                        (B,D) movie latents
        Z_lat_tgts: Tensor                (B,L,D)
        Yp_tgts_batched: list[Tensor]     per-person-field targets, each (B,L,...)
        t_grid: Tensor                    (B,L)
        time_mask: Tensor                 (B,L)
    """
    if len(batch) == 0:
        raise ValueError("Empty batch passed to collate_batch")

    # movie fields
    num_m_fields = len(batch[0]["Mx"])
    Mx_batched: List[torch.Tensor] = []
    My_batched: List[torch.Tensor] = []

    for fi in range(num_m_fields):
        xs = [b["Mx"][fi] for b in batch]
        ys = [b["My"][fi] for b in batch]
        Mx_batched.append(torch.stack(xs, dim=0))
        My_batched.append(torch.stack(ys, dim=0))

    # movie latents
    Zt = torch.stack([b["movie_latent"] for b in batch], dim=0)

    # latent trajectories
    Z_lat_tgts = torch.stack([b["Z_lat_tgts"] for b in batch], dim=0)

    # per-person fields
    num_p_fields = len(batch[0]["Yp_tgts"])
    Yp_tgts_batched: List[torch.Tensor] = []
    for fi in range(num_p_fields):
        ys = [b["Yp_tgts"][fi] for b in batch]
        Yp_tgts_batched.append(torch.stack(ys, dim=0))

    # time grid + mask
    t_grid = torch.stack([b["t_grid"] for b in batch], dim=0)
    time_mask = torch.stack([b["time_mask"] for b in batch], dim=0)

    return Mx_batched, My_batched, Zt, Z_lat_tgts, Yp_tgts_batched, t_grid, time_mask
