from typing import List, Tuple, Dict, Optional
from collections import OrderedDict
import sqlite3
import torch
from torch.utils.data import IterableDataset

class _LRU:
    def __init__(self, capacity: int = 200000):
        self.capacity = int(capacity)
        self.d = OrderedDict()

    def get(self, k):
        v = self.d.get(k)
        if v is None:
            return None
        self.d.move_to_end(k)
        return v

    def set(self, k, v):
        self.d[k] = v
        self.d.move_to_end(k)
        if len(self.d) > self.capacity:
            self.d.popitem(last=False)

class TitlePathIterable(IterableDataset):
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
    ):
        super().__init__()
        self.db_path = db_path
        self.principals_table = principals_table
        self.movie_ae = movie_ae
        self.people_ae = people_ae
        self.num_people = int(num_people)
        self.shuffle = bool(shuffle)
        self.movie_limit = None if movie_limit is None else int(movie_limit)
        self.seed = int(seed)

        self._conn: Optional[sqlite3.Connection] = None
        self._cur: Optional[sqlite3.Cursor] = None
        self._scan_cur: Optional[sqlite3.Cursor] = None

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

    def _open(self):
        if self._conn is not None:
            return
        conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA temp_store = MEMORY;");
        conn.execute("PRAGMA cache_size = -200000;")
        conn.execute("PRAGMA mmap_size = 268435456;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        self._conn = conn
        self._cur = conn.cursor()
        self._scan_cur = conn.cursor()

    def _close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = None
        self._cur = None
        self._scan_cur = None

    def _iter_tconsts(self):
        if self.movie_limit is not None:
            it = self._scan_cur.execute(
                """
                SELECT tconst
                FROM (
                    SELECT DISTINCT tconst, (movie_hash + ?) AS key
                    FROM edges
                )
                ORDER BY key
                LIMIT ?
                """,
                (self.seed, self.movie_limit),
            )
            for r in it:
                yield r[0]
            return

        if self.shuffle:
            it = self._scan_cur.execute(
                "SELECT DISTINCT tconst FROM edges ORDER BY RANDOM()"
            )
        else:
            it = self._scan_cur.execute(
                "SELECT DISTINCT tconst FROM edges"
            )
        for r in it:
            yield r[0]

    def _movie_row(self, tconst: str) -> Dict:
        r = self._movie_cache.get(tconst)
        if r is not None:
            return r
        row = self._cur.execute(self._movie_sql, (tconst, tconst)).fetchone()
        if row is None:
            out = {
                "tconst": tconst,
                "primaryTitle": None,
                "startYear": None,
                "endYear": None,
                "runtimeMinutes": None,
                "averageRating": None,
                "numVotes": None,
                "genres": [],
            }
            self._movie_cache.set(tconst, out)
            return out
        out = {
            "tconst": tconst,
            "primaryTitle": row[0],
            "startYear": row[1],
            "endYear": row[2],
            "runtimeMinutes": row[3],
            "averageRating": row[4],
            "numVotes": row[5],
            "genres": row[6].split(",") if row[6] else [],
        }
        self._movie_cache.set(tconst, out)
        return out

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

    @torch.no_grad()
    def _encode_movie(self, row_dict) -> torch.Tensor:
        key = row_dict.get("tconst")
        if key is not None:
            cached = self._z_movie_cache.get(key)
            if cached is not None:
                return cached
        xs = [f.transform(row_dict.get(f.name)).unsqueeze(0).to(self.movie_ae.device) for f in self.movie_ae.fields]
        z = self.movie_ae.encoder(xs).detach().cpu().squeeze(0)
        if key is not None:
            self._z_movie_cache.set(key, z)
        return z

    @torch.no_grad()
    def _encode_person(self, nconst: str):
        c = self._people_cache.get(nconst)
        if c is None:
            row = self._cur.execute(
                """
                SELECT p.primaryName, p.birthYear, p.deathYear,
                       (SELECT GROUP_CONCAT(profession,',') FROM people_professions WHERE nconst = p.nconst)
                FROM people p
                WHERE p.nconst = ? LIMIT 1
                """,
                (nconst,),
            ).fetchone()
            if row is None:
                row_dict = {"primaryName": None, "birthYear": None, "deathYear": None, "professions": None}
            else:
                row_dict = {
                    "primaryName": row[0],
                    "birthYear": row[1],
                    "deathYear": row[2],
                    "professions": row[3].split(",") if row[3] else None,
                }
            xs = [f.transform(row_dict.get(f.name)) for f in self.people_ae.fields]
            ys = [f.transform_target(row_dict.get(f.name)) for f in self.people_ae.fields]
            self._people_cache.set(nconst, (xs, ys))
        else:
            xs, ys = c

        zc = self._z_person_cache.get(nconst)
        if zc is None:
            X = [x.unsqueeze(0).to(self.people_ae.device) for x in xs]
            z = self.people_ae.encoder(X).detach().cpu().squeeze(0)
            self._z_person_cache.set(nconst, z)
        else:
            z = zc
        return xs, ys, z

    def __iter__(self):
        self._open()
        try:
            for tconst in self._iter_tconsts():
                mrow = self._movie_row(tconst)
                Mx = [f.transform(mrow.get(f.name)) for f in self.movie_ae.fields]
                My = [f.transform_target(mrow.get(f.name)) for f in self.movie_ae.fields]
                z_title = self._encode_movie(mrow)

                nconsts = self._people_for_title(tconst)
                people_k = len(nconsts)

                # fixed grid length (title + N people)
                L_target = self.num_people + 1
                t_grid = torch.linspace(0.0, 1.0, steps=L_target)

                # mask: 1 for title and actual people, else 0
                valid_len = min(people_k + 1, L_target)
                time_mask = torch.zeros(L_target, dtype=torch.float32)
                time_mask[:valid_len] = 1.0

                per_field_targets: List[List[torch.Tensor]] = [[] for _ in self.people_ae.fields]
                z_steps: List[torch.Tensor] = [z_title]

                for i in range(people_k):
                    px, py, z = self._encode_person(nconsts[i])
                    for fi, y in enumerate(py):
                        per_field_targets[fi].append(y)
                    z_steps.append(z)

                # prepend a dummy at t=0 for each person field
                for fi in range(len(per_field_targets)):
                    dummy0 = self.people_ae.fields[fi].get_base_padding_value()
                    seq = [dummy0] + per_field_targets[fi]
                    # pad to fixed length with dummies when not enough people
                    if len(seq) < L_target:
                        pad_needed = L_target - len(seq)
                        seq += [dummy0] * pad_needed
                    per_field_targets[fi] = seq

                # z targets: title + people; if fewer, pad by repeating last valid latent (masked anyway)
                if len(z_steps) < L_target:
                    last = z_steps[-1]
                    z_steps += [last] * (L_target - len(z_steps))
                Z_latent_targets = torch.stack(z_steps, dim=0)

                yield Mx, My, z_title, Z_latent_targets, per_field_targets, t_grid, time_mask
        finally:
            self._close()

def _pad_along_time(x_list: List[torch.Tensor], max_L: int, pad_like: torch.Tensor) -> torch.Tensor:
    parts = []
    for x in x_list:
        L = x.size(0)
        if L == max_L:
            parts.append(x.unsqueeze(0))
            continue
        pad = pad_like.unsqueeze(0).expand(max_L - L, *pad_like.shape)
        parts.append(torch.cat([x, pad], dim=0).unsqueeze(0))
    return torch.cat(parts, dim=0)

def collate_batch(batch):
    Mx_cols = list(zip(*[b[0] for b in batch]))
    My_cols = list(zip(*[b[1] for b in batch]))
    Mx = [torch.stack(col, dim=0) for col in Mx_cols]
    My = [torch.stack(col, dim=0) for col in My_cols]

    Z_titles = torch.stack([b[2] for b in batch], dim=0)

    # sequences are already fixed to (1+N); keep padding logic for safety
    seq_lens = [b[3].size(0) for b in batch]
    max_L = max(seq_lens)
    D = batch[0][3].size(-1)

    Z_targets_list = []
    t_list = []
    m_list = []
    for _, _, _, z_seq, _, t_seq, mask_seq in batch:
        L = z_seq.size(0)
        if L < max_L:
            pad_z = torch.zeros(max_L - L, D, dtype=z_seq.dtype, device=z_seq.device)
            z_padded = torch.cat([z_seq, pad_z], dim=0)
            tail_t = t_seq.new_full((max_L - L,), float(t_seq[-1].item()))
            t_padded = torch.cat([t_seq, tail_t], dim=0)
            m_padded = torch.cat([mask_seq, torch.zeros(max_L - L, dtype=mask_seq.dtype, device=mask_seq.device)], dim=0)
        else:
            z_padded = z_seq
            t_padded = t_seq
            m_padded = mask_seq
        Z_targets_list.append(z_padded.unsqueeze(0))
        t_list.append(t_padded.unsqueeze(0))
        m_list.append(m_padded.unsqueeze(0))

    Z_targets = torch.cat(Z_targets_list, dim=0)
    t_grid = torch.cat(t_list, dim=0)
    mask = torch.cat(m_list, dim=0)

    people_field_targets: List[torch.Tensor] = []
    num_fields = len(batch[0][4])
    for fi in range(num_fields):
        per_sample_time_lists = [b[4][fi] for b in batch]
        pad_like = None
        for lst in per_sample_time_lists:
            if isinstance(lst, (list, tuple)) and len(lst) > 0:
                pad_like = lst[0]
                break
        if pad_like is None:
            pad_like = torch.zeros((), dtype=torch.float32)

        per_sample_stacked = []
        for lst in per_sample_time_lists:
            if isinstance(lst, (list, tuple)):
                if len(lst) == 0:
                    per_sample_stacked.append(pad_like.new_zeros((0,) + pad_like.shape))
                else:
                    per_sample_stacked.append(torch.stack(lst, dim=0))
            else:
                per_sample_stacked.append(lst)

        stacked = _pad_along_time(per_sample_stacked, max_L, pad_like)
        people_field_targets.append(stacked)

    return Mx, My, Z_titles, Z_targets, people_field_targets, t_grid, mask
