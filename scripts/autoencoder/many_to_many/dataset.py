# scripts/autoencoder/many_to_many/dataset.py
from __future__ import annotations
import json
import sqlite3
from typing import List, Tuple, Dict, Optional, Iterator
import torch
from torch.utils.data import IterableDataset
from scripts.autoencoder.fields import BaseField
from scripts.sql_filters import (
    movie_from_join, movie_where_clause, movie_group_by,
    people_from_join, people_where_clause, people_group_by, people_having,
)

class ManyToManyDataset(IterableDataset):
    def __init__(
        self,
        db_path: str,
        movie_fields: List[BaseField],
        people_fields: List[BaseField],
        seq_len_titles: int,
        seq_len_people: int,
        movie_limit: Optional[int] = None,
        person_limit: Optional[int] = None,
        mode_ratio: Optional[float] = None,
    ):
        super().__init__()
        self.db_path = db_path
        self.movie_fields = movie_fields
        self.people_fields = people_fields
        self.seq_len_titles = int(seq_len_titles)
        self.seq_len_people = int(seq_len_people)
        self.movie_limit = movie_limit
        self.person_limit = person_limit
        self._mode_ratio = mode_ratio

    def _count_movies(self, con) -> int:
        q = f"""
        SELECT COUNT(*) FROM (
            SELECT t.tconst {movie_from_join()}
            WHERE {movie_where_clause()}
            {movie_group_by()}
        ) x
        """
        return int(con.execute(q).fetchone()[0])

    def _count_people(self, con) -> int:
        q = f"""
        SELECT COUNT(*) FROM (
            SELECT p.nconst {people_from_join()}
            WHERE {people_where_clause()}
            {people_group_by()}
            {people_having()}
        ) x
        """
        return int(con.execute(q).fetchone()[0])

    def _movie_rows(self, con) -> Iterator[Dict]:
        q = f"""
        SELECT t.tconst, t.primaryTitle, t.startYear
        {movie_from_join()}
        WHERE {movie_where_clause()}
        {movie_group_by()}
        """
        if self.movie_limit:
            q += f" LIMIT {int(self.movie_limit)}"
        for tconst, title, startYear in con.execute(q):
            yield {"tconst": tconst, "primaryTitle": title, "startYear": startYear}

    def _people_rows(self, con) -> Iterator[Dict]:
        q = f"""
        SELECT p.nconst, p.primaryName, p.birthYear
        {people_from_join()}
        WHERE {people_where_clause()}
        {people_group_by()}
        {people_having()}
        """
        if self.person_limit:
            q += f" LIMIT {int(self.person_limit)}"
        for nconst, name, birthYear in con.execute(q):
            yield {"nconst": nconst, "primaryName": name, "birthYear": birthYear}

    def _people_for_movie(self, con, tconst: str, limit: int) -> List[Dict]:
        q = f"""
        SELECT p.primaryName, p.birthYear, p.deathYear, GROUP_CONCAT(pp.profession, ',')
        {people_from_join()} INNER JOIN principals pr ON pr.nconst = p.nconst
        WHERE pr.tconst = ? AND {people_where_clause()}
        {people_group_by()}
        {people_having()}
        ORDER BY pr.ordering
        LIMIT ?
        """
        out: List[Dict] = []
        for n, b, d, profs in con.execute(q, (tconst, int(limit))):
            out.append({
                "primaryName": n,
                "birthYear": b,
                "deathYear": d,
                "professions": profs.split(",") if profs else None,
            })
        return out

    def _titles_for_person(self, con, nconst: str, limit: int) -> List[Dict]:
        q1 = """
        SELECT t.primaryTitle, t.startYear
        FROM people_known_for k
        JOIN titles t ON t.tconst = k.tconst
        WHERE k.nconst = ?
        LIMIT ?
        """
        out: List[Dict] = []
        for title, year in con.execute(q1, (nconst, int(limit))):
            out.append({"primaryTitle": title, "startYear": year})
        if out:
            return out
        q2 = """
        SELECT t.primaryTitle, t.startYear
        FROM principals pr
        JOIN titles t ON t.tconst = pr.tconst
        WHERE pr.nconst = ?
        ORDER BY pr.ordering
        LIMIT ?
        """
        for title, year in con.execute(q2, (nconst, int(limit))):
            out.append({"primaryTitle": title, "startYear": year})
        return out

    def _transform_seq(self, fields: List[BaseField], rows: List[Dict], seq_len: int) -> List[torch.Tensor]:
        ys: List[torch.Tensor] = []
        for f in fields:
            steps = []
            for i in range(seq_len):
                if i < len(rows) and rows[i] is not None:
                    steps.append(f.transform_target(rows[i].get(f.name)))
                else:
                    steps.append(f.get_base_padding_value())
            ys.append(torch.stack(steps, dim=0))
        return ys

    def _pad_source_seq(self, fields: List[BaseField], src_row: Dict, seq_len: int) -> List[torch.Tensor]:
        xs: List[torch.Tensor] = []
        for f in fields:
            steps = []
            for i in range(seq_len):
                if i == 0 and src_row is not None:
                    steps.append(f.transform(src_row.get(f.name)))
                else:
                    steps.append(f.get_base_padding_value())
            xs.append(torch.stack(steps, dim=0))
        return xs

    def __iter__(self):
        con = sqlite3.connect(self.db_path, check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute("PRAGMA temp_store=MEMORY;")
        con.execute("PRAGMA cache_size=-200000;")
        con.execute("PRAGMA mmap_size=268435456;")
        con.execute("PRAGMA busy_timeout=5000;")

        num_movies = self._count_movies(con)
        num_people = self._count_people(con)
        denom = max(1, num_movies + num_people)
        r_movies = float(num_movies) / float(denom)
        r_people = float(num_people) / float(denom)
        want_movie_mode = self._mode_ratio if self._mode_ratio is not None else r_movies

        it_movies = self._movie_rows(con)
        it_people = self._people_rows(con)

        for m_row in it_movies:
            ppl = self._people_for_movie(con, m_row["tconst"], self.seq_len_people)
            orig_len = min(len(ppl), self.seq_len_people)
            if orig_len == 0:
                continue
            if orig_len < self.seq_len_people:
                ppl = ppl + [{} for _ in range(self.seq_len_people - orig_len)]
            else:
                ppl = ppl[: self.seq_len_people]

            xm = self._pad_source_seq(self.movie_fields, m_row, self.seq_len_titles)
            xp = self._pad_source_seq(self.people_fields, None, self.seq_len_people)
            yt = self._transform_seq(self.movie_fields, [], self.seq_len_titles)
            yp = self._transform_seq(self.people_fields, ppl, self.seq_len_people)

            mt = torch.zeros(self.seq_len_titles, dtype=torch.float32)
            mp = torch.zeros(self.seq_len_people, dtype=torch.float32)
            mp[:orig_len] = 1.0

            mode = torch.tensor(0, dtype=torch.long)
            yield xm, xp, yt, yp, mt, mp, mode

        for p_row in it_people:
            titles = self._titles_for_person(con, p_row["nconst"], self.seq_len_titles)
            orig_len = min(len(titles), self.seq_len_titles)
            if orig_len == 0:
                continue
            if orig_len < self.seq_len_titles:
                titles = titles + [{} for _ in range(self.seq_len_titles - orig_len)]
            else:
                titles = titles[: self.seq_len_titles]

            xm = self._pad_source_seq(self.movie_fields, None, self.seq_len_titles)
            xp = self._pad_source_seq(self.people_fields, p_row, self.seq_len_people)
            yt = self._transform_seq(self.movie_fields, titles, self.seq_len_titles)
            yp = self._transform_seq(self.people_fields, [], self.seq_len_people)

            mt = torch.zeros(self.seq_len_titles, dtype=torch.float32)
            mp = torch.zeros(self.seq_len_people, dtype=torch.float32)
            mt[:orig_len] = 1.0

            mode = torch.tensor(1, dtype=torch.long)
            yield xm, xp, yt, yp, mt, mp, mode

        con.close()


def collate_many_to_many(batch):
    xm, xp, yt, yp, mt, mp, mode = zip(*batch)
    xm_cols = list(zip(*xm))
    xp_cols = list(zip(*xp))
    yt_cols = list(zip(*yt))
    yp_cols = list(zip(*yp))
    Xm = [torch.stack(col, dim=0) for col in xm_cols]
    Xp = [torch.stack(col, dim=0) for col in xp_cols]
    Yt = [torch.stack(col, dim=0) for col in yt_cols]
    Yp = [torch.stack(col, dim=0) for col in yp_cols]
    Mt = torch.stack(mt, dim=0)
    Mp = torch.stack(mp, dim=0)
    Mode = torch.stack(mode, dim=0)
    return Xm, Xp, Yt, Yp, Mt, Mp, Mode
