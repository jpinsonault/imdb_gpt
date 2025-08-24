import json
import sqlite3
from pathlib import Path

def build_movie_people_seq(db_path: str, seq_len: int):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")
    conn.execute("PRAGMA mmap_size=268435456;")
    conn.execute("PRAGMA busy_timeout=5000;")

    conn.execute("""
    CREATE TABLE IF NOT EXISTS movie_people_seq (
        tconst TEXT PRIMARY KEY,
        people_json TEXT
    );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_movie_people_seq_tconst ON movie_people_seq(tconst);")

    movie_sql = """
    SELECT t.tconst
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

    people_sql = """
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

    cur = conn.cursor()
    cur.execute(movie_sql)
    ins = conn.cursor()

    batch = []
    for (tconst,) in cur:
        rows = conn.execute(people_sql, (tconst, seq_len)).fetchall()
        if not rows:
            continue
        plist = []
        for r in rows:
            profs = r[3].split(",") if r[3] else []
            plist.append({
                "primaryName": r[0],
                "birthYear": r[1],
                "deathYear": r[2],
                "professions": profs
            })
        batch.append((tconst, json.dumps(plist)))
        if len(batch) >= 1000:
            ins.executemany("INSERT OR REPLACE INTO movie_people_seq (tconst, people_json) VALUES (?,?)", batch)
            conn.commit()
            batch.clear()

    if batch:
        ins.executemany("INSERT OR REPLACE INTO movie_people_seq (tconst, people_json) VALUES (?,?)", batch)
        conn.commit()

    conn.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    args = parser.parse_args()
    build_movie_people_seq(args.db, args.seq_len)
