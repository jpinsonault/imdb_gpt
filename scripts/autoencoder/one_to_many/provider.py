# scripts/autoencoder/one_to_many/provider.py
from __future__ import annotations
from typing import Iterator, List, Dict, Optional
import sqlite3

from scripts.sql_filters import movie_from_join, movie_group_by, movie_where_clause, people_from_join, people_group_by, people_having, people_where_clause

class OneToManyProvider:
    def iter_sources(self) -> Iterator[Dict]:
        raise NotImplementedError

    def targets_for(
        self,
        source_row: Dict,
        limit: int,
    ) -> List[Dict]:
        raise NotImplementedError

    @property
    def seq_len(self) -> int:
        raise NotImplementedError


class ImdbMovieToPeopleProvider(OneToManyProvider):
    def __init__(
        self,
        db_path: str,
        seq_len: int,
        movie_limit: Optional[int] = None,
    ):
        self.db_path = db_path
        self._seq_len = int(seq_len)
        self.movie_limit = movie_limit
        self.conn = None

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def _ensure(self):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)

    def iter_sources(self) -> Iterator[Dict]:
        self._ensure()
        sql = f"""
        SELECT t.tconst, t.primaryTitle, t.startYear
        {movie_from_join()}
        WHERE
            {movie_where_clause()}
        {movie_group_by()}
        """
        if self.movie_limit:
            sql += f" LIMIT {int(self.movie_limit)}"
        for tconst, title, startYear in self.conn.execute(sql):
            yield {
                "tconst": tconst,
                "primaryTitle": title,
                "startYear": startYear,
            }


    def targets_for(self, source_row: Dict, limit: int) -> List[Dict]:
        self._ensure()
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
        for n, b, d, profs in self.conn.execute(q, (source_row.get("tconst"), int(limit))):
            out.append({
                "primaryName": n,
                "birthYear": b,
                "deathYear": d,
                "professions": profs.split(",") if profs else None,
            })
        return out



class ImdbPeopleToMovieProvider(OneToManyProvider):
    def __init__(
        self,
        db_path: str,
        seq_len: int,
        person_limit: Optional[int] = None,
    ):
        self.db_path = db_path
        self._seq_len = int(seq_len)
        self.person_limit = person_limit
        self.conn = None

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def _ensure(self):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)

    def iter_sources(self) -> Iterator[Dict]:
        self._ensure()
        sql = f"""
        SELECT p.nconst, p.primaryName, p.birthYear
        {people_from_join()}
        WHERE
            {people_where_clause()}
        {people_group_by()}
        {people_having()}
        """
        if self.person_limit:
            sql += f" LIMIT {int(self.person_limit)}"
        for nconst, name, birthYear in self.conn.execute(sql):
            yield {
                "nconst": nconst,
                "primaryName": name,
                "birthYear": birthYear,
            }


    def targets_for(
        self,
        source_row: Dict,
        limit: int,
    ) -> List[Dict]:
        self._ensure()
        q = """
        SELECT t.primaryTitle, t.startYear
        FROM people_known_for k
        JOIN titles t ON t.tconst = k.tconst
        WHERE k.nconst = ?
        LIMIT ?
        """
        out: List[Dict] = []
        for title, year in self.conn.execute(q, (source_row.get("nconst"), int(limit))):
            out.append({
                "primaryTitle": title,
                "startYear": year,
            })
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
        for title, year in self.conn.execute(q2, (source_row.get("nconst"), int(limit))):
            out.append({
                "primaryTitle": title,
                "startYear": year,
            })
        return out
