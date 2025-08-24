# scripts/autoencoder/sequence_datasets.py
import json
import sqlite3
from typing import List
import torch
from torch.utils.data import IterableDataset
from .fields import BaseField

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
            t.endYear,
            t.runtimeMinutes,
            t.averageRating,
            t.numVotes,
            GROUP_CONCAT(g.genre, ',')
        FROM titles t
        INNER JOIN title_genres g ON g.tconst = t.tconst
        WHERE
            t.startYear IS NOT NULL
            AND t.averageRating IS NOT NULL
            AND t.runtimeMinutes IS NOT NULL
            AND t.runtimeMinutes >= 5
            AND t.startYear >= 1850
            AND t.titleType IN ('movie','tvSeries','tvMovie','tvMiniSeries')
            AND t.numVotes >= 10
        GROUP BY t.tconst
        """
        self.seq_lookup_sql = "SELECT people_json FROM movie_people_seq WHERE tconst = ? LIMIT 1"

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
            tconst, title, startYear, endYear, runtime, rating, votes, genres = row
            s = con.execute(self.seq_lookup_sql, (tconst,)).fetchone()
            if not s:
                continue
            ppl = json.loads(s[0]) or []
            orig_len = min(len(ppl), self.seq_len)
            if orig_len < self.seq_len:
                ppl = ppl + [{} for _ in range(self.seq_len - orig_len)]
            else:
                ppl = ppl[: self.seq_len]

            movie_row = {
                "tconst": tconst,
                "primaryTitle": title,
                "startYear": startYear,
                "endYear": endYear,
                "runtimeMinutes": runtime,
                "averageRating": rating,
                "numVotes": votes,
                "genres": genres.split(",") if genres else [],
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
            t.endYear,
            t.runtimeMinutes,
            t.averageRating,
            t.numVotes,
            GROUP_CONCAT(g.genre, ',')
        FROM titles t
        INNER JOIN title_genres g ON g.tconst = t.tconst
        WHERE
            t.startYear IS NOT NULL
            AND t.averageRating IS NOT NULL
            AND t.runtimeMinutes IS NOT NULL
            AND t.runtimeMinutes >= 5
            AND t.startYear >= 1850
            AND t.titleType IN ('movie','tvSeries','tvMovie','tvMiniSeries')
            AND t.numVotes >= 10
        GROUP BY t.tconst
        """
        self.people_sql = """
        SELECT
            p.primaryName,
            p.birthYear,
            p.deathYear,
            GROUP_CONCAT(pp.profession, ',')
        FROM people p
        LEFT JOIN people_professions pp ON p.nconst = pp.nconst
        INNER JOIN principals pr ON pr.nconst = p.nconst
        WHERE pr.tconst = ? AND p.birthYear IS NOT NULL
        GROUP BY p.nconst
        HAVING COUNT(pp.profession) > 0
        ORDER BY pr.ordering
        LIMIT ?
        """
        self.samples = self._materialize()

    def _materialize(self):
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
        for tconst, title, startYear, endYear, runtime, rating, votes, genres in cur:
            movie_row = {
                "tconst": tconst,
                "primaryTitle": title,
                "startYear": startYear,
                "endYear": endYear,
                "runtimeMinutes": runtime,
                "averageRating": rating,
                "numVotes": votes,
                "genres": genres.split(",") if genres else [],
            }
            ppl_rows = con.execute(self.people_sql, (tconst, self.seq_len)).fetchall()
            if not ppl_rows:
                continue
            ppl = []
            for r in ppl_rows:
                ppl.append({
                    "primaryName": r[0],
                    "birthYear": r[1],
                    "deathYear": r[2],
                    "professions": r[3].split(",") if r[3] else None,
                })
            orig_len = min(len(ppl), self.seq_len)
            if orig_len < self.seq_len:
                ppl = ppl + [{} for _ in range(self.seq_len - orig_len)]
            else:
                ppl = ppl[: self.seq_len]
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
