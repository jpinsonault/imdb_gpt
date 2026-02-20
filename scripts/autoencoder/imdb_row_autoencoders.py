# scripts/autoencoder/imdb_row_autoencoders.py

from typing import List, Dict
import sqlite3
import logging
import torch
import math

from scripts.sql_filters import (
    movie_from_join, 
    movie_group_by, 
    movie_having, 
    movie_where_clause, 
    people_from_join, 
    people_group_by, 
    people_having, 
    people_where_clause,
    movie_select_clause,
    people_select_clause,
    map_movie_row,
    map_person_row
)

from .fields import (
    NumericDigitCategoryField,
    TextField,
    MultiCategoryField,
    ScalarField,
    Scaling,
    BaseField,
)
from .row_autoencoder import RowAutoencoder

class TitlesAutoencoder(RowAutoencoder):
    def build_fields(self) -> list[BaseField]:
        return [
            NumericDigitCategoryField("tconst", strip_nonnumeric=True),
            TextField("primaryTitle", max_length=32),
            NumericDigitCategoryField("startYear"),
            NumericDigitCategoryField("runtimeMinutes"),
            NumericDigitCategoryField("averageRating", fraction_digits=1),
            NumericDigitCategoryField("numVotes"),
            MultiCategoryField("genres"),
            NumericDigitCategoryField("castCount"),
            NumericDigitCategoryField("directorCount"),
            NumericDigitCategoryField("writerCount"),
        ]

    def accumulate_stats(self):
        """
        Optimized SQL-based stats accumulation.
        """
        if self.config.use_cache and self._load_cache():
            logging.info("[Titles] stats loaded from cache")
            return

        logging.info("[Titles] Fast-accumulating stats via SQL...")
        
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            where_sql = movie_where_clause()

            # 1. SCALARS & DIGITS: Combined Pass
            # We fetch all min/max values in ONE query. 
            # This is 5x faster than individual queries as it only scans the index/table once.
            logging.info("[Titles] Aggregating scalar ranges...")
            q_scalars = f"""
            SELECT 
                MIN(t.startYear), MAX(t.startYear),
                MIN(t.endYear), MAX(t.endYear),
                MIN(t.runtimeMinutes), MAX(t.runtimeMinutes),
                MIN(t.averageRating), MAX(t.averageRating),
                MIN(t.numVotes), MAX(t.numVotes)
            FROM titles t 
            WHERE {where_sql}
            """
            row = cur.execute(q_scalars).fetchone()
            
            # Unpack results safely (handle None if DB is empty)
            if row and row[0] is not None:
                self._manual_inject_digits("startYear", row[0], row[1])
                self._manual_inject_digits("runtimeMinutes", row[4], row[5])
                self._manual_inject_digits("averageRating", row[6], row[7])
                self._manual_inject_digits("numVotes", row[8], row[9])
            else:
                logging.warning("[Titles] No data found matching filter!")

            # 2. peopleCount-ish upper bound (reuse for per-head counts)
            # We use the new index on `principals(tconst)`. 
            # We assume the max people count won't exceed 5000 to save compute, 
            # or we calculate it efficiently.
            logging.info("[Titles] Calculating max people per title...")
            
            logging.info("[Titles] Calculating max peopleCount...")
            q_pc = f"""
            SELECT MAX(cnt) FROM (
                SELECT COUNT(pr.ordering) as cnt 
                FROM principals pr 
                WHERE pr.tconst IN (SELECT tconst FROM titles t WHERE {where_sql})
                GROUP BY pr.tconst
            )
            """
            mx_pc = cur.execute(q_pc).fetchone()
            mx_pc_val = mx_pc[0] if mx_pc and mx_pc[0] else 50

            # Use the same upper bound for each per-head count.
            self._manual_inject_digits("castCount", 0, mx_pc_val)
            self._manual_inject_digits("directorCount", 0, mx_pc_val)
            self._manual_inject_digits("writerCount", 0, mx_pc_val)

            # 3. CATEGORICAL: Genres
            logging.info("[Titles] Accumulating Genres...")
            # Query strictly on existence.
            q_genres = f"""
            SELECT DISTINCT g.genre 
            FROM title_genres g
            WHERE g.tconst IN (SELECT tconst FROM titles t WHERE {where_sql} LIMIT 50000)
            """
            # Optimization: We limit the inner subquery sample. 
            # It is highly unlikely a genre appears only in the tail of the dataset.
            for (genre,) in cur.execute(q_genres):
                self.get_field("genres").accumulate_stats([genre])

            # 4. TEXT: Sampling
            logging.info("[Titles] Sampling text for tokenizer...")
            # ORDER BY RANDOM() on full table is slow. 
            # We grab a block of rows using LIMIT/OFFSET via rowid if possible, 
            # or just limit the scan.
            q_text = f"""
            SELECT t.primaryTitle FROM titles t
            WHERE {where_sql}
            LIMIT 100000
            """
            f_text = self.get_field("primaryTitle")
            for (txt,) in cur.execute(q_text):
                f_text.accumulate_stats(txt)

            # 5. TCONST
            self._manual_inject_digits("tconst", 0, 99999999)

        self.finalize_stats()

    def _manual_inject_digits(self, field_name, min_val, max_val):
        f = self.get_field(field_name)
        if min_val is None: min_val = 0
        if max_val is None: max_val = 0
        f.accumulate_stats(min_val)
        f.accumulate_stats(max_val)

    def get_field(self, name) -> BaseField:
        for f in self.fields:
            if f.name == name: return f
        raise KeyError(name)

    def row_generator(self):
        import sqlite3
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            c = conn.cursor()
            c.execute("PRAGMA cache_size = -100000;") 
            c.execute(
                f"""
                SELECT 
                    {movie_select_clause(alias='t', genre_alias='g')}
                {movie_from_join()}
                WHERE 
                    {movie_where_clause()}
                {movie_group_by()}
                {movie_having()}
                LIMIT ?
                """,
                (self.config.movie_limit,)
            )
            for row in c:
                yield map_movie_row(row)

    def row_by_tconst(self, tconst: str) -> dict:
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            c = conn.cursor()
            c.execute(
                f"""
                SELECT 
                    {movie_select_clause(alias='t', genre_alias='g')}
                FROM titles t
                LEFT JOIN title_genres g ON g.tconst = t.tconst
                WHERE t.tconst = ?
                GROUP BY t.tconst
            """,
                (tconst,),
            )
            r = c.fetchone()
            if r is None:
                raise KeyError(tconst)
        return map_movie_row(r)


class PeopleAutoencoder(RowAutoencoder):
    def build_fields(self) -> List[BaseField]:
        return [
            NumericDigitCategoryField("nconst", strip_nonnumeric=True),
            TextField("primaryName", max_length=32),
            NumericDigitCategoryField("birthYear"),
            NumericDigitCategoryField("deathYear"),
            MultiCategoryField("professions"),
            NumericDigitCategoryField("castCount"),
            NumericDigitCategoryField("directorCount"),
            NumericDigitCategoryField("writerCount"),
        ]

    def accumulate_stats(self):
        if self.config.use_cache and self._load_cache():
            logging.info("[People] stats loaded from cache")
            return

        logging.info("[People] Fast-accumulating stats via SQL...")
        
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            where_sql = people_where_clause()

            # 1. SCALARS: Combined
            logging.info("[People] Aggregating scalar ranges...")
            q_scalars = f"""
            SELECT 
                MIN(p.birthYear), MAX(p.birthYear),
                MIN(p.deathYear), MAX(p.deathYear)
            FROM people p
            WHERE {where_sql}
            """
            row = cur.execute(q_scalars).fetchone()
            if row and row[0] is not None:
                self._manual_inject_digits("birthYear", row[0], row[1])
                self._manual_inject_digits("deathYear", row[2], row[3])

            # 2. titleCount-ish upper bound for per-head counts
            # Optimized to use Index
            logging.info("[People] Calculating max titleCount...")
            q_tc = f"""
            SELECT MAX(cnt) FROM (
                SELECT COUNT(pr.tconst) as cnt 
                FROM principals pr
                WHERE pr.nconst IN (SELECT nconst FROM people p WHERE {where_sql} LIMIT 100000)
                GROUP BY pr.nconst
            )
            """
            mx_tc = cur.execute(q_tc).fetchone()
            if mx_tc and mx_tc[0]:
                max_titles = mx_tc[0]
            else:
                max_titles = 200

            self._manual_inject_digits("castCount", 0, max_titles)
            self._manual_inject_digits("directorCount", 0, max_titles)
            self._manual_inject_digits("writerCount", 0, max_titles)

            # 3. Professions
            logging.info("[People] Accumulating Professions...")
            q_prof = f"""
            SELECT DISTINCT pp.profession 
            FROM people_professions pp
            """
            f_prof = self.get_field("professions")
            for (prof,) in cur.execute(q_prof):
                f_prof.accumulate_stats([prof])

            # 4. Text Sampling
            logging.info("[People] Sampling names...")
            q_text = f"""
            SELECT p.primaryName FROM people p
            WHERE {where_sql}
            LIMIT 100000
            """
            f_text = self.get_field("primaryName")
            for (txt,) in cur.execute(q_text):
                f_text.accumulate_stats(txt)

            # nconst
            self._manual_inject_digits("nconst", 0, 99999999)

        # Do NOT set self.stats_accumulated = True here manually.
        self.finalize_stats()

    def _manual_inject_digits(self, field_name, min_val, max_val):
        f = self.get_field(field_name)
        if min_val is None: min_val = 0
        if max_val is None: max_val = 0
        f.accumulate_stats(min_val)
        f.accumulate_stats(max_val)

    def get_field(self, name) -> BaseField:
        for f in self.fields:
            if f.name == name: return f
        raise KeyError(name)

    def row_generator(self):
        import sqlite3
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            c = conn.cursor()
            c.execute("PRAGMA cache_size = -100000;")
            c.execute(
                f"""
                SELECT 
                    {people_select_clause(alias='p', prof_alias='pp')}
                {people_from_join()}
                WHERE 
                    {people_where_clause()}
                {people_group_by()}
                {people_having()}
                """
            )
            for row in c:
                yield map_person_row(row)

    def row_by_nconst(self, nconst: str) -> dict:
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            c = conn.cursor()
            c.execute(
                f"""
                SELECT 
                    {people_select_clause(alias='p', prof_alias='pp')}
                FROM people p
                LEFT JOIN people_professions pp ON pp.nconst = p.nconst
                WHERE p.nconst = ?
                GROUP BY p.nconst
            """,
                (nconst,),
            )
            r = c.fetchone()
            if r is None:
                raise KeyError(nconst)
        return map_person_row(r)
