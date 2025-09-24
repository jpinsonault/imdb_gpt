import json
import sqlite3
from collections import OrderedDict
from typing import List, Dict, Tuple, Iterator
import torch
from torch.utils.data import IterableDataset
from scripts.autoencoder.fields import BaseField

class OneToManyDataset(IterableDataset):
    def __init__(
        self,
        db_path: str,
        source_fields: List[BaseField],
        target_fields: List[BaseField],
        seq_len: int,
        movie_limit: int | None = None,
        movie_cache_size: int = 10000,
    ):
        super().__init__()
        self.db_path = db_path
        self.source_fields = source_fields
        self.target_fields = target_fields
        self.seq_len = int(seq_len)
        self.movie_limit = movie_limit
        self.movie_cache_size = int(movie_cache_size)
        self._movie_cache: OrderedDict[str, Tuple[torch.Tensor, ...]] = OrderedDict()
        self._sql = """
        SELECT
            t.tconst,
            t.primaryTitle,
            t.startYear,
            s.people_json
        FROM movie_people_seq s
        JOIN titles t ON t.tconst = s.tconst
        """

    def _cache_put(self, key: str, val: Tuple[torch.Tensor, ...]):
        if key in self._movie_cache:
            self._movie_cache.move_to_end(key)
            return
        self._movie_cache[key] = val
        if len(self._movie_cache) > self.movie_cache_size:
            self._movie_cache.popitem(last=False)

    def _cache_get(self, key: str):
        v = self._movie_cache.get(key)
        if v is None:
            return None
        self._movie_cache.move_to_end(key)
        return v

    def __iter__(self):
        con = sqlite3.connect(self.db_path, check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute("PRAGMA temp_store=MEMORY;")
        con.execute("PRAGMA cache_size=-200000;")
        con.execute("PRAGMA mmap_size=268435456;")
        con.execute("PRAGMA busy_timeout=5000;")

        cur = con.cursor()
        lim = "" if self.movie_limit is None else f" LIMIT {int(self.movie_limit)}"
        cur.execute(self._sql + lim)

        for tconst, title, startYear, people_json in cur:
            ppl = json.loads(people_json) or []
            orig_len = min(len(ppl), self.seq_len)
            if orig_len == 0:
                continue
            if orig_len < self.seq_len:
                ppl = ppl + [{} for _ in range(self.seq_len - orig_len)]
            else:
                ppl = ppl[: self.seq_len]

            cached = self._cache_get(tconst)
            if cached is None:
                movie_row: Dict[str, object] = {
                    "tconst": tconst,
                    "primaryTitle": title,
                    "startYear": startYear,
                }
                x_source = tuple(f.transform(movie_row.get(f.name)) for f in self.source_fields)
                self._cache_put(tconst, x_source)
            else:
                x_source = cached

            y_targets: List[torch.Tensor] = []
            for f in self.target_fields:
                steps = []
                for i in range(self.seq_len):
                    if i < orig_len:
                        steps.append(f.transform_target(ppl[i].get(f.name)))
                    else:
                        steps.append(f.get_base_padding_value())
                y_targets.append(torch.stack(steps, dim=0))

            mask = torch.zeros(self.seq_len, dtype=torch.float32)
            mask[:orig_len] = 1.0
            yield list(x_source), y_targets, mask

        con.close()


def collate_one_to_many(batch):
    xm_cols = list(zip(*[b[0] for b in batch]))
    yp_cols = list(zip(*[b[1] for b in batch]))
    Xm = [torch.stack(col, dim=0) for col in xm_cols]
    Yp = [torch.stack(col, dim=0) for col in yp_cols]
    M = torch.stack([b[2] for b in batch], dim=0)
    return Xm, Yp, M


class ProviderBackedOneToManyDataset(IterableDataset):
    def __init__(
        self,
        provider,
        source_fields: List[BaseField],
        target_fields: List[BaseField],
        seq_len: int,
    ):
        super().__init__()
        self.provider = provider
        self.source_fields = source_fields
        self.target_fields = target_fields
        self.seq_len = int(seq_len)

    def __iter__(self) -> Iterator[Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]]:
        for src in self.provider.iter_sources():
            tgt_rows = self.provider.targets_for(src, self.seq_len)
            orig_len = min(len(tgt_rows), self.seq_len)
            if orig_len == 0:
                continue
            if orig_len < self.seq_len:
                pad = [{} for _ in range(self.seq_len - orig_len)]
                tgt_rows = tgt_rows + pad
            else:
                tgt_rows = tgt_rows[: self.seq_len]

            x_source = [f.transform(src.get(f.name)) for f in self.source_fields]
            y_targets: List[torch.Tensor] = []
            for f in self.target_fields:
                steps = []
                for i in range(self.seq_len):
                    if i < orig_len:
                        steps.append(f.transform_target(tgt_rows[i].get(f.name)))
                    else:
                        steps.append(f.get_base_padding_value())
                y_targets.append(torch.stack(steps, dim=0))

            mask = torch.zeros(self.seq_len, dtype=torch.float32)
            mask[:orig_len] = 1.0
            yield x_source, y_targets, mask
