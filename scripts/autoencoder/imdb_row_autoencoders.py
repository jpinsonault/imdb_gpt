# scripts/autoencoder/imdb_row_autoencoders.py

from typing import List
import sqlite3

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
            TextField("primaryTitle"),
            NumericDigitCategoryField("startYear"),
            NumericDigitCategoryField("endYear"),
            NumericDigitCategoryField("runtimeMinutes"),
            NumericDigitCategoryField("averageRating"),
            NumericDigitCategoryField("numVotes"),
            MultiCategoryField("genres"),
            # Renamed from principalCount to peopleCount
            NumericDigitCategoryField("peopleCount"),
        ]

    def row_generator(self):
        import sqlite3
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            c = conn.cursor()
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
                (self.config.movie_limit,),
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
            TextField("primaryName"),
            NumericDigitCategoryField("birthYear"),
            NumericDigitCategoryField("deathYear"),
            MultiCategoryField("professions"),
            NumericDigitCategoryField("titleCount"),
        ]

    def row_generator(self):
        import sqlite3
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            c = conn.cursor()
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