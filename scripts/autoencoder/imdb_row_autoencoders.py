from pathlib import Path
from typing import Any, Dict, List
import sqlite3

from autoencoder.fields import (
    NumericDigitCategoryField,
    TextField,
    MultiCategoryField,
    BaseField
)

from autoencoder.row_autoencoder import RowAutoencoder

class TitlesAutoencoder(RowAutoencoder):
    def build_fields(self) -> list[BaseField]:
        return [
            TextField("primaryTitle"),
            NumericDigitCategoryField("startYear"),
            # NumericDigitCategoryField("runtimeMinutes"),
            # NumericDigitCategoryField("averageRating", fraction_digits=1),
            # NumericDigitCategoryField("numVotes"),
            # MultiCategoryField("genres"),
        ]

    def row_generator(self):
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            c = conn.cursor()
            c.execute("""
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
                HAVING COUNT(g.genre) > 0
                LIMIT ?
            """, (self.config["movie_limit"],))
            for tconst, primaryTitle, startYear, endYear, runtime, rating, votes, genres in c:
                yield {
                    "tconst":        tconst,
                    "primaryTitle":  primaryTitle,
                    "startYear":     startYear,
                    "endYear":       endYear,
                    "runtimeMinutes": runtime,
                    "averageRating": rating,
                    "numVotes":      votes,
                    "genres":        genres.split(","),
                }

    def row_by_tconst(self, tconst: str) -> dict:
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT
                    t.primaryTitle,
                    t.startYear,
                    t.endYear,
                    t.runtimeMinutes,
                    t.averageRating,
                    t.numVotes,
                    GROUP_CONCAT(g.genre, ',')
                FROM titles t
                LEFT JOIN title_genres g ON g.tconst = t.tconst
                WHERE t.tconst = ?
                GROUP BY t.tconst
            """, (tconst,))
            r = c.fetchone()
            if r is None:
                raise KeyError(tconst)
        return {
            "tconst":        tconst,
            "primaryTitle":  r[0],
            "startYear":     r[1],
            "endYear":       r[2],
            "runtimeMinutes":r[3],
            "averageRating": r[4],
            "numVotes":      r[5],
            "genres":        r[6].split(",") if r[6] else [],
        }



class PeopleAutoencoder(RowAutoencoder):
    def build_fields(self) -> List[BaseField]:
        return [
            TextField("primaryName"),
            NumericDigitCategoryField("birthYear"),
            # NumericDigitCategoryField("deathYear", optional=True),
            # MultiCategoryField("professions")
        ]

    def row_generator(self):
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT
                    p.primaryName,
                    p.birthYear,
                    p.deathYear,
                    GROUP_CONCAT(pp.profession, ',') AS professions
                FROM people p
                LEFT JOIN people_professions pp ON p.nconst = pp.nconst
                WHERE p.birthYear IS NOT NULL
                GROUP BY p.nconst
                HAVING COUNT(pp.profession) > 0
            """)
            for row in c:
                yield {
                    "primaryName": row[0],
                    "birthYear": row[1],
                    "deathYear": row[2],
                    "professions": row[3].split(',') if row[3] else None
                }

    def row_by_nconst(self, nconst: str) -> dict:
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT
                    p.primaryName,
                    p.birthYear,
                    p.deathYear,
                    GROUP_CONCAT(pp.profession, ',')
                FROM people p
                LEFT JOIN people_professions pp ON pp.nconst = p.nconst
                WHERE p.nconst = ?
                GROUP BY p.nconst
            """, (nconst,))
            r = c.fetchone()
            if r is None:
                raise KeyError(nconst)
        return {
            "primaryName": r[0],
            "birthYear":   r[1],
            "deathYear":   r[2],
            "professions": r[3].split(',') if r[3] else None,
        }