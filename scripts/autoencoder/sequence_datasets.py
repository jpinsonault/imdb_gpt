# scripts/autoencoder/sequence_datasets.py
import json
import sqlite3
from typing import List
import torch
from torch.utils.data import IterableDataset
from .fields import BaseField
import logging
from tqdm import tqdm
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MoviesPeopleSequenceDataset(IterableDataset):
    def __init__(
        self,
        db_path: str,
        movie_fields: List[BaseField],
        people_fields: List[BaseField],
        seq_len: int,
        movie_limit: int | None = None,
    ):
        super().__init__()
        self.db_path = db_path
        self.movie_fields = movie_fields
        self.people_fields = people_fields
        self.seq_len = seq_len
        self.movie_limit = movie_limit
        self.movie_sql = """
        SELECT
            t.tconst,
            t.primaryTitle,
            t.startYear,
            s.people_json
        FROM movie_people_seq s
        JOIN titles t ON t.tconst = s.tconst
        """
        # movie_people_seq already encodes all the quality filters up-front

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
        cur.execute(self.movie_sql + lim)

        for row in cur:
            tconst, title, startYear, people_json = row
            ppl = json.loads(people_json) or []
            orig_len = min(len(ppl), self.seq_len)
            if orig_len < self.seq_len:
                ppl = ppl + [{} for _ in range(self.seq_len - orig_len)]
            else:
                ppl = ppl[: self.seq_len]

            movie_row = {
                "tconst": tconst,
                "primaryTitle": title,
                "startYear": startYear,
            }

            x_movie = [f.transform(movie_row.get(f.name)) for f in self.movie_fields]

            y_people = []
            for f in self.people_fields:
                steps = []
                for t in range(self.seq_len):
                    if t < orig_len:
                        steps.append(f.transform_target(ppl[t].get(f.name)))
                    else:
                        steps.append(f.get_base_padding_value())
                y_people.append(torch.stack(steps, dim=0))

            mask = torch.zeros(self.seq_len, dtype=torch.float32)
            mask[:orig_len] = 1.0
            yield x_movie, y_people, mask

        con.close()


def _collate(batch):
    xm_cols = list(zip(*[b[0] for b in batch]))
    yp_cols = list(zip(*[b[1] for b in batch]))
    Xm = [torch.stack(col, dim=0) for col in xm_cols]
    Yp = [torch.stack(col, dim=0) for col in yp_cols]
    M = torch.stack([b[2] for b in batch], dim=0)
    return Xm, Yp, M


class MoviesPeopleSequenceMemoryDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        db_path: str,
        movie_fields: List[BaseField],
        people_fields: List[BaseField],
        seq_len: int,
        movie_limit: int | None = None,
    ):
        super().__init__()
        self.db_path = db_path
        self.movie_fields = movie_fields
        self.people_fields = people_fields
        self.seq_len = seq_len
        self.movie_limit = movie_limit
        self.movie_sql = """
        SELECT
            t.tconst,
            t.primaryTitle,
            t.startYear,
            s.people_json
        FROM movie_people_seq s
        JOIN titles t ON t.tconst = s.tconst
        """
        self.samples = self._materialize()

    def _materialize(self):
        logging.info("Materializing dataset...")
        con = sqlite3.connect(self.db_path, check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute("PRAGMA temp_store=MEMORY;")
        con.execute("PRAGMA cache_size=-200000;")
        con.execute("PRAGMA mmap_size=268435456;")
        con.execute("PRAGMA busy_timeout=5000;")
        cur = con.cursor()
        lim = "" if self.movie_limit is None else f" LIMIT {int(self.movie_limit)}"
        cur.execute(self.movie_sql + lim)
        out = []
        for tconst, title, startYear, people_json in tqdm(cur, desc="Loading rows", unit=" rows"):
            ppl = json.loads(people_json) or []
            if not ppl:
                continue
            orig_len = min(len(ppl), self.seq_len)
            if orig_len < self.seq_len:
                ppl = ppl + [{} for _ in range(self.seq_len - orig_len)]
            else:
                ppl = ppl[: self.seq_len]
            movie_row = {
                "tconst": tconst,
                "primaryTitle": title,
                "startYear": startYear,
            }
            x_movie = [f.transform(movie_row.get(f.name)) for f in self.movie_fields]
            y_people = []
            for f in self.people_fields:
                steps = []
                for t in range(self.seq_len):
                    if t < orig_len:
                        steps.append(f.transform_target(ppl[t].get(f.name)))
                    else:
                        steps.append(f.get_base_padding_value())
                y_people.append(torch.stack(steps, dim=0))
            mask = torch.zeros(self.seq_len, dtype=torch.float32)
            mask[:orig_len] = 1.0
            out.append((x_movie, y_people, mask))
        con.close()
        return out

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]
