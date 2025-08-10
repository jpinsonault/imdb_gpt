import sqlite3
from typing import Dict, Iterator, List, Tuple, Optional

def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")
    conn.execute("PRAGMA mmap_size=268435456;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn

MOVIE_SQL = """
SELECT
    t.tconst, t.primaryTitle, t.startYear, t.endYear,
    t.runtimeMinutes, t.averageRating, t.numVotes,
    (SELECT GROUP_CONCAT(genre, ',') FROM title_genres g WHERE g.tconst = t.tconst)
FROM titles t
WHERE
    t.startYear IS NOT NULL
    AND t.averageRating IS NOT NULL
    AND t.runtimeMinutes IS NOT NULL
    AND t.runtimeMinutes >= 5
    AND t.startYear >= 1850
    AND t.titleType IN ('movie','tvSeries','tvMovie','tvMiniSeries')
    AND t.numVotes >= 10
"""

PEOPLE_FOR_MOVIE_SQL = """
SELECT
    p.primaryName, p.birthYear, p.deathYear,
    (SELECT GROUP_CONCAT(profession, ',') FROM people_professions pp WHERE pp.nconst = p.nconst)
FROM people p
INNER JOIN principals pr ON pr.nconst = p.nconst
WHERE pr.tconst = ? AND p.birthYear IS NOT NULL
GROUP BY p.nconst
HAVING COUNT(1) > 0
ORDER BY pr.ordering
LIMIT ?
"""

def iter_movie_with_people(db_path: str, seq_len: int) -> Iterator[Tuple[Dict, List[Dict]]]:
    conn = _connect(db_path)
    cur_m = conn.cursor()
    cur_p = conn.cursor()
    cur_m.execute(MOVIE_SQL)
    for row in cur_m:
        tconst, primaryTitle, startYear, endYear, runtime, rating, votes, genres_str = row
        m_row = {
            "primaryTitle": primaryTitle,
            "startYear": startYear,
            "genres": genres_str.split(",") if genres_str else [],
        }
        ppl: List[Dict] = []
        cur_p.execute(PEOPLE_FOR_MOVIE_SQL, (tconst, seq_len))
        for pn, by, dy, profs in cur_p.fetchall():
            ppl.append({"primaryName": pn, "birthYear": by})
        if not ppl:
            continue
        if len(ppl) < seq_len:
            ppl = ppl + [ppl[-1]] * (seq_len - len(ppl))
        else:
            ppl = ppl[:seq_len]
        yield m_row, ppl
    conn.close()
