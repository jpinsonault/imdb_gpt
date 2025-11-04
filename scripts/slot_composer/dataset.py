from typing import List, Tuple, Iterator, Dict, Optional
from collections import OrderedDict
import sqlite3
import torch
from torch.utils.data import IterableDataset

class _LRU:
    def __init__(self, capacity: int = 100000):
        self.capacity = int(capacity)
        self.d = OrderedDict()

    def get(self, key):
        v = self.d.get(key)
        if v is None:
            return None
        self.d.move_to_end(key)
        return v

    def set(self, key, value):
        self.d[key] = value
        self.d.move_to_end(key)
        if len(self.d) > self.capacity:
            self.d.popitem(last=False)


class TitlePeopleIterable(IterableDataset):
    def __init__(
        self,
        db_path: str,
        principals_table: str,
        movie_ae,
        people_ae,
        num_slots: int,
        people_cache_capacity: int = 200000,
    ):
        super().__init__()
        self.db_path = db_path
        self.principals_table = principals_table
        self.movie_ae = movie_ae
        self.people_ae = people_ae
        self.num_slots = int(num_slots)
        self.people_cache_capacity = int(people_cache_capacity)

        self._conn: Optional[sqlite3.Connection] = None
        self._cur: Optional[sqlite3.Cursor] = None
        self._people_cache = _LRU(self.people_cache_capacity)

    def _open(self):
        if self._conn is not None:
            return
        conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA temp_store = MEMORY;")
        conn.execute("PRAGMA cache_size = -200000;")
        conn.execute("PRAGMA mmap_size = 268435456;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        self._conn = conn
        self._cur = conn.cursor()

    def _close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = None
        self._cur = None

    def _people_for_title(self, tconst: str) -> List[str]:
        rows = self._cur.execute(
            f"""
            SELECT pr.nconst
            FROM {self.principals_table} pr
            JOIN people p ON p.nconst = pr.nconst
            WHERE pr.tconst = ?
              AND p.birthYear IS NOT NULL
            ORDER BY pr.ordering ASC
            LIMIT ?
            """,
            (tconst, self.num_slots),
        ).fetchall()
        return [r[0] for r in rows]

    def _bulk_people_rows(self, nconsts: List[str]) -> Dict[str, Dict]:
        out: Dict[str, Dict] = {}
        need: List[str] = []
        for n in nconsts:
            r = self._people_cache.get(n)
            if r is not None:
                out[n] = r
            else:
                need.append(n)
        if not need:
            return out

        placeholders = ",".join(["?"] * len(need))
        rows = self._cur.execute(
            f"""
            SELECT p.nconst,
                   p.primaryName,
                   p.birthYear,
                   p.deathYear,
                   GROUP_CONCAT(pp.profession, ',')
            FROM people p
            LEFT JOIN people_professions pp ON pp.nconst = p.nconst
            WHERE p.nconst IN ({placeholders})
            GROUP BY p.nconst
            """,
            need,
        ).fetchall()

        for nconst, name, by, dy, profs in rows:
            row = {
                "primaryName": name,
                "birthYear": by,
                "deathYear": dy,
                "professions": profs.split(",") if profs else None,
            }
            self._people_cache.set(nconst, row)
            out[nconst] = row

        # for any missing ids (rare), fill minimal rows to avoid key errors
        for n in need:
            if n not in out:
                row = {"primaryName": None, "birthYear": None, "deathYear": None, "professions": None}
                self._people_cache.set(n, row)
                out[n] = row

        return out

    def __iter__(self) -> Iterator[
        Tuple[
            List[torch.Tensor],
            List[torch.Tensor],
            List[List[torch.Tensor]],
            List[List[torch.Tensor]],
            torch.Tensor,
        ]
    ]:
        self._open()
        try:
            for row in self.movie_ae.row_generator():
                tconst = row.get("tconst")
                m_x = [f.transform(row.get(f.name)) for f in self.movie_ae.fields]
                m_y = [f.transform_target(row.get(f.name)) for f in self.movie_ae.fields]

                nconsts = self._people_for_title(tconst)
                k = len(nconsts)

                p_rows = self._bulk_people_rows(nconsts)

                p_x: List[List[torch.Tensor]] = []
                p_y: List[List[torch.Tensor]] = []
                for i in range(self.num_slots):
                    if i < k:
                        prow = p_rows.get(nconsts[i], None)
                        if prow is None:
                            px = [f.get_base_padding_value() for f in self.people_ae.fields]
                            py = [f.get_base_padding_value() for f in self.people_ae.fields]
                        else:
                            px = [f.transform(prow.get(f.name)) for f in self.people_ae.fields]
                            py = [f.transform_target(prow.get(f.name)) for f in self.people_ae.fields]
                    else:
                        px = [f.get_base_padding_value() for f in self.people_ae.fields]
                        py = [f.get_base_padding_value() for f in self.people_ae.fields]
                    p_x.append(px)
                    p_y.append(py)

                mask = torch.zeros(self.num_slots, dtype=torch.float32)
                if k > 0:
                    mask[:min(k, self.num_slots)] = 1.0

                yield m_x, m_y, p_x, p_y, mask
        finally:
            self._close()


def collate_batch(
    batch: List[
        Tuple[
            List[torch.Tensor],
            List[torch.Tensor],
            List[List[torch.Tensor]],
            List[List[torch.Tensor]],
            torch.Tensor,
        ]
    ]
):
    m_x_cols = list(zip(*[b[0] for b in batch]))
    m_y_cols = list(zip(*[b[1] for b in batch]))
    Mx = [torch.stack(col, dim=0) for col in m_x_cols]
    My = [torch.stack(col, dim=0) for col in m_y_cols]

    num_slots = batch[0][4].numel()
    pf = len(batch[0][2][0])
    Pxs = []
    Pys = []
    for f in range(pf):
        slot_tensors_x = []
        slot_tensors_y = []
        for i in range(num_slots):
            bx = [b[2][i][f] for b in batch]
            by = [b[3][i][f] for b in batch]
            slot_tensors_x.append(torch.stack(bx, dim=0))
            slot_tensors_y.append(torch.stack(by, dim=0))
        Pxs.append(torch.stack(slot_tensors_x, dim=1))
        Pys.append(torch.stack(slot_tensors_y, dim=1))

    mask = torch.stack([b[4] for b in batch], dim=0)
    return Mx, My, Pxs, Pys, mask
