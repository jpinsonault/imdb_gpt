import sqlite3
from typing import Dict, Iterator, List, Optional, Tuple

def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")
    conn.execute("PRAGMA mmap_size=268435456;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn

def iter_titles(db_path: str, movie_limit: int) -> Iterator[Dict]:
    sql = """
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
    """
    with _connect(db_path) as conn:
        for row in conn.execute(sql, (movie_limit,)):
            tconst, primaryTitle, startYear, endYear, runtime, rating, votes, genres = row
            yield {
                "tconst": tconst,
                "primaryTitle": primaryTitle,
                "startYear": startYear,
                "endYear": endYear,
                "runtimeMinutes": runtime,
                "averageRating": rating,
                "numVotes": votes,
                "genres": genres.split(",") if genres else [],
            }

def get_title_by_tconst(db_path: str, tconst: str) -> Dict:
    sql = """
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
    """
    with _connect(db_path) as conn:
        r = conn.execute(sql, (tconst,)).fetchone()
        if r is None:
            raise KeyError(tconst)
        return {
            "tconst": tconst,
            "primaryTitle": r[0],
            "startYear": r[1],
            "endYear": r[2],
            "runtimeMinutes": r[3],
            "averageRating": r[4],
            "numVotes": r[5],
            "genres": r[6].split(",") if r[6] else [],
        }

def iter_people(db_path: str) -> Iterator[Dict]:
    sql = """
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
    """
    with _connect(db_path) as conn:
        for row in conn.execute(sql):
            yield {
                "primaryName": row[0],
                "birthYear": row[1],
                "deathYear": row[2],
                "professions": row[3].split(",") if row[3] else None,
            }

def get_person_by_nconst(db_path: str, nconst: str) -> Dict:
    sql = """
    SELECT
        p.primaryName,
        p.birthYear,
        p.deathYear,
        GROUP_CONCAT(pp.profession, ',')
    FROM people p
    LEFT JOIN people_professions pp ON pp.nconst = p.nconst
    WHERE p.nconst = ?
    GROUP BY p.nconst
    """
    with _connect(db_path) as conn:
        r = conn.execute(sql, (nconst,)).fetchone()
        if r is None:
            raise KeyError(nconst)
        return {
            "primaryName": r[0],
            "birthYear": r[1],
            "deathYear": r[2],
            "professions": r[3].split(",") if r[3] else None,
        }
