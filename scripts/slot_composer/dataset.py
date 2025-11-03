from typing import List, Tuple, Iterator
import sqlite3
import torch
from torch.utils.data import IterableDataset

class TitlePeopleIterable(IterableDataset):
    def __init__(
        self,
        db_path: str,
        principals_table: str,
        movie_ae,
        people_ae,
        num_slots: int,
    ):
        super().__init__()
        self.db_path = db_path
        self.principals_table = principals_table
        self.movie_ae = movie_ae
        self.people_ae = people_ae
        self.num_slots = int(num_slots)

    def _people_for_title(self, cur, tconst: str) -> List[str]:
        rows = cur.execute(
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

    def __iter__(self) -> Iterator[
        Tuple[
            List[torch.Tensor],
            List[torch.Tensor],
            List[List[torch.Tensor]],
            List[List[torch.Tensor]],
            torch.Tensor,
        ]
    ]:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = conn.cursor()
        try:
            for row in self.movie_ae.row_generator():
                tconst = row.get("tconst")
                m_x = [f.transform(row.get(f.name)) for f in self.movie_ae.fields]
                m_y = [f.transform_target(row.get(f.name)) for f in self.movie_ae.fields]

                nconsts = self._people_for_title(cur, tconst)
                k = len(nconsts)

                p_x: List[List[torch.Tensor]] = []
                p_y: List[List[torch.Tensor]] = []
                for i in range(self.num_slots):
                    if i < k:
                        prow = self.people_ae.row_by_nconst(nconsts[i])
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
            conn.close()

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
