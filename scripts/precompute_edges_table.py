# scripts/precompute_edges_table.py
import sqlite3, hashlib, random, sys
from pathlib import Path
from tqdm import tqdm
from config import project_config


##############################################################################
# Utility
##############################################################################
def _hash32(txt: str) -> int:
    return int(hashlib.md5(txt.encode()).hexdigest()[:8], 16)


##############################################################################
# Step 1 – create / reset the table
##############################################################################
def create_edges_table(conn: sqlite3.Connection):
    print("\n[1/4] Creating empty `edges` table …")
    conn.execute("DROP TABLE IF EXISTS edges;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS edges (
            edgeId      INTEGER PRIMARY KEY AUTOINCREMENT,
            tconst      TEXT NOT NULL,
            nconst      TEXT NOT NULL,
            movie_hash  INTEGER,
            person_hash INTEGER,
            UNIQUE(tconst, nconst) ON CONFLICT IGNORE
        );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_tconst  ON edges(tconst);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_nconst  ON edges(nconst);")
    conn.commit()
    print("    ✓ table & indices ready")


##############################################################################
# Step 2 – sample‑based estimate
##############################################################################
def estimate_edges(conn: sqlite3.Connection, sample_size=1000, seed=1234):
    print("\n[2/4] Estimating edges from a 100‑movie sample …")

    all_titles = [r[0] for r in conn.execute("SELECT tconst FROM titles WHERE startYear IS NOT NULL;")]
    total_movies = len(all_titles)
    print(f"      total movies: {total_movies:,}")

    random.seed(seed)
    sample = random.sample(all_titles, k=min(sample_size, total_movies))

    counts = []
    cur = conn.cursor()
    for t in tqdm(sample, desc="      counting", unit="movie", leave=False):
        num_people = cur.execute("""
            SELECT COUNT(DISTINCT pr.nconst)
            FROM principals pr
            JOIN people p ON p.nconst = pr.nconst
            WHERE pr.tconst = ?
              AND p.birthYear IS NOT NULL
        """, (t,)).fetchone()[0]
        counts.append(num_people)

    avg = sum(counts) / len(counts)
    est_edges = int(round(avg * total_movies))

    print(f"        • sample avg  : {avg:.2f} people/movie")
    print(f"        • est. edges  : {est_edges:,}\n")
    return est_edges


##############################################################################
# Step 3 – stream edges & insert in chunks
##############################################################################
def stream_and_insert_edges(conn: sqlite3.Connection, chunk_size=10_000):
    print("[3/4] Streaming edges and inserting (Ctrl-C to abort safely) …")

    cur_read  = conn.cursor()
    cur_write = conn.cursor()

    query = """
        SELECT pr.tconst, pr.nconst
        FROM   principals pr
        JOIN   titles   t ON t.tconst  = pr.tconst
        JOIN   people   p ON p.nconst  = pr.nconst
        WHERE  t.startYear IS NOT NULL
          AND  p.birthYear IS NOT NULL
    """

    cur_read.execute(query)

    bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

    batch = []
    total_inserted = 0
    try:
        for tconst, nconst in tqdm(cur_read, unit="edge", bar_format=bar_format):
            batch.append((tconst, nconst, _hash32(tconst), _hash32(nconst)))
            if len(batch) == chunk_size:
                cur_write.executemany(
                    "INSERT INTO edges (tconst, nconst, movie_hash, person_hash) VALUES (?,?,?,?);",
                    batch
                )
                conn.commit()
                total_inserted += len(batch)
                batch.clear()

        # final flush
        if batch:
            cur_write.executemany(
                "INSERT INTO edges (tconst, nconst, movie_hash, person_hash) VALUES (?,?,?,?);",
                batch
            )
            conn.commit()
            total_inserted += len(batch)

    except KeyboardInterrupt:
        print("\n!! Interrupted by user – committing current chunk …")
        if batch:
            cur_write.executemany(
                "INSERT INTO edges (tconst, nconst, movie_hash, person_hash) VALUES (?,?,?,?);",
                batch
            )
            conn.commit()
            total_inserted += len(batch)
        print(f"      ✓ partial insert committed ({total_inserted:,} edges).")
        sys.exit(0)

    print(f"      ✓ finished – {total_inserted:,} edges inserted.")


##############################################################################
# Main
##############################################################################
def main():
    db_path = Path(project_config["data_dir"]) / "imdb.db"
    conn = sqlite3.connect(db_path)

    create_edges_table(conn)
    estimate_edges(conn, sample_size=100, seed=1234)
    stream_and_insert_edges(conn, chunk_size=10_000)

    conn.close()
    print("\nAll done – `edges` table is ready.\n")


if __name__ == "__main__":
    main()
