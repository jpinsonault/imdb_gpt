# scripts/precompute_edges_table.py
"""
Populate an `edges` table that only contains (movie tconst, person nconst)
pairs for rows that pass the **same quality filters** as the movie & people
autoencoders.  In particular we now require:

    • non‑NULL   t.averageRating
    • non‑NULL   t.runtimeMinutes   AND  t.runtimeMinutes ≥ 5
    • non‑NULL   t.startYear        AND  t.startYear ≥ 1850
    •            t.titleType ∈ {'movie','tvSeries','tvMovie','tvMiniSeries'}
    •            t.numVotes ≥ 10
    • at least   1 genre           (INNER JOIN title_genres)
    • non‑NULL   p.birthYear
"""

from __future__ import annotations
import sqlite3, hashlib, random, sys
from pathlib import Path
from tqdm import tqdm
from config import project_config


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _hash32(txt: str) -> int:
    """Fast deterministic 32‑bit hash for shardable integer keys"""
    return int(hashlib.md5(txt.encode(), usedforsecurity=False).hexdigest()[:8], 16)


# --------------------------------------------------------------------------- #
# [1] blank / reset the table
# --------------------------------------------------------------------------- #
def create_edges_table(conn: sqlite3.Connection) -> None:
    print("\n[1/4] Creating empty `edges` table …")
    conn.execute("DROP TABLE IF EXISTS edges;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS edges (
            edgeId       INTEGER PRIMARY KEY AUTOINCREMENT,
            tconst       TEXT NOT NULL,
            nconst       TEXT NOT NULL,
            movie_hash   INTEGER,
            person_hash  INTEGER,
            UNIQUE(tconst, nconst) ON CONFLICT IGNORE
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_tconst  ON edges(tconst);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_nconst  ON edges(nconst);")
    conn.commit()
    print("    ✓ table & indices ready")


# --------------------------------------------------------------------------- #
# [2] estimate row count (optional, just to show progress)
# --------------------------------------------------------------------------- #
MOVIE_FILTER_CLAUSE = """
    t.startYear IS NOT NULL
    AND t.startYear >= 1850
    AND t.averageRating IS NOT NULL
    AND t.runtimeMinutes IS NOT NULL
    AND t.runtimeMinutes >= 5
    AND t.titleType IN ('movie','tvSeries','tvMovie','tvMiniSeries')
    AND t.numVotes >= 10
"""

def estimate_edges(conn: sqlite3.Connection, sample_size: int = 1_000, *, seed: int = 1234) -> int:
    print("\n[2/4] Estimating edges from a movie sample …")

    # candidate movies that pass **all** filters (same as TitlesAutoencoder)
    all_titles = [
        r[0]
        for r in conn.execute(
            f"""
            SELECT t.tconst
            FROM titles t
            INNER JOIN title_genres g ON g.tconst = t.tconst
            WHERE {MOVIE_FILTER_CLAUSE}
            GROUP BY t.tconst
            HAVING COUNT(g.genre) > 0
            """
        )
    ]

    total_movies = len(all_titles)
    print(f"      movies passing filters: {total_movies:,}")

    if not total_movies:
        print("      (no candidate movies ‑‑ returning estimate 0)")
        return 0

    random.seed(seed)
    sample = random.sample(all_titles, k=min(sample_size, total_movies))

    counts: list[int] = []
    cur = conn.cursor()
    for t in tqdm(sample, desc="      sampling", unit="movie", leave=False):
        num_people = cur.execute(
            """
            SELECT COUNT(DISTINCT pr.nconst)
            FROM principals pr
            JOIN people p ON p.nconst = pr.nconst
            WHERE pr.tconst = ?
              AND p.birthYear IS NOT NULL
            """,
            (t,),
        ).fetchone()[0]
        counts.append(num_people)

    avg = sum(counts) / len(counts)
    est_edges = int(round(avg * total_movies))

    print(f"        • sample avg : {avg:.2f} people/movie")
    print(f"        • est. edges : {est_edges:,}\n")
    return est_edges


# --------------------------------------------------------------------------- #
# [3] stream through principals & insert
# --------------------------------------------------------------------------- #
INSERT_CHUNK = 10_000

def stream_and_insert_edges(conn: sqlite3.Connection, *, chunk_size: int = INSERT_CHUNK) -> None:
    print("[3/4] Running edge insertion …")

    read_cur  = conn.cursor()
    write_cur = conn.cursor()

    # The heavy lifting query – mirrors Titles/PeopleAutoencoder filters
    read_cur.execute(
        f"""
        SELECT pr.tconst,
               pr.nconst
        FROM   principals        pr
        JOIN   titles            t  ON t.tconst  = pr.tconst
        JOIN   title_genres      g  ON g.tconst  = t.tconst   -- ensures ≥1 genre
        JOIN   people            p  ON p.nconst  = pr.nconst
        WHERE  {MOVIE_FILTER_CLAUSE}
          AND  p.birthYear IS NOT NULL
        GROUP BY pr.tconst, pr.nconst     -- collapse duplicates from multi‑genres
        """
    )

    bar_format = "{l_bar}{bar}| {n_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

    batch: list[tuple[str, str, int, int]] = []
    total_inserted = 0

    try:
        for tconst, nconst in tqdm(read_cur, unit="edge", bar_format=bar_format):
            batch.append((tconst, nconst, _hash32(tconst), _hash32(nconst)))
            if len(batch) == chunk_size:
                write_cur.executemany(
                    "INSERT INTO edges (tconst, nconst, movie_hash, person_hash) VALUES (?,?,?,?);",
                    batch,
                )
                conn.commit()
                total_inserted += len(batch)
                batch.clear()

        # final flush
        if batch:
            write_cur.executemany(
                "INSERT INTO edges (tconst, nconst, movie_hash, person_hash) VALUES (?,?,?,?);",
                batch,
            )
            conn.commit()
            total_inserted += len(batch)

    except KeyboardInterrupt:
        print("\n!! Interrupted – committing current chunk …")
        if batch:
            write_cur.executemany(
                "INSERT INTO edges (tconst, nconst, movie_hash, person_hash) VALUES (?,?,?,?);",
                batch,
            )
            conn.commit()
            total_inserted += len(batch)
        print(f"      ✓ partial commit ({total_inserted:,} edges) done.")
        sys.exit(0)

    print(f"      ✓ finished – {total_inserted:,} edges inserted.")


# --------------------------------------------------------------------------- #
# main CLI helper
# --------------------------------------------------------------------------- #
def main() -> None:
    db_path = Path(project_config.data_dir) / "imdb.db"
    conn = sqlite3.connect(db_path)

    create_edges_table(conn)
    estimate_edges(conn, sample_size=1_000, seed=1234)
    stream_and_insert_edges(conn, chunk_size=INSERT_CHUNK)

    conn.close()
    print("\n[4/4] All done – `edges` table ready.\n")


if __name__ == "__main__":
    main()
