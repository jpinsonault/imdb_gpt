# scripts/precompute_edges_table.py
from __future__ import annotations
import sqlite3
import hashlib
import sys
from pathlib import Path
from tqdm import tqdm
from config import project_config
from scripts.sql_filters import movie_where_clause

# --------------------------------------------------------------------------- #
# Helpers & Setup
# --------------------------------------------------------------------------- #

def _hash32(txt: str) -> int:
    """Fast deterministic 32‑bit hash for shardable integer keys."""
    if txt is None:
        return 0
    return int(hashlib.md5(txt.encode(), usedforsecurity=False).hexdigest()[:8], 16)

def register_adapters(conn: sqlite3.Connection):
    """
    Register the python hash function as a SQL function so we can 
    compute hashes entirely inside the DB engine (much faster).
    """
    conn.create_function("calc_hash", 1, _hash32)

# --------------------------------------------------------------------------- #
# [1] Schema Setup
# --------------------------------------------------------------------------- #

def create_edges_table(conn: sqlite3.Connection) -> None:
    print("\n[1/5] Creating `edges` schema...")
    conn.execute("DROP TABLE IF EXISTS edges;")
    conn.execute(
        """
        CREATE TABLE edges (
            edgeId        INTEGER PRIMARY KEY AUTOINCREMENT,
            tconst        TEXT NOT NULL,
            nconst        TEXT NOT NULL,
            movie_hash    INTEGER,
            person_hash   INTEGER,
            UNIQUE(tconst, nconst) ON CONFLICT IGNORE
        );
        """
    )
    # We delay index creation until AFTER insertion for max speed
    conn.commit()

# --------------------------------------------------------------------------- #
# [2 & 3] Pre-Filtering (The Speed Boost)
# --------------------------------------------------------------------------- #

def create_valid_filters(conn: sqlite3.Connection):
    """
    Creates temporary tables for valid movies and people.
    This avoids re-scanning the massive text tables during the join.
    """
    print("[2/5] Pre-filtering Valid Movies...")
    
    # We use the centralized filter logic
    movie_filter = movie_where_clause()
    
    conn.execute("DROP TABLE IF EXISTS temp_valid_movies")
    
    # Create a simple list of tconsts that pass all criteria
    conn.execute(f"""
        CREATE TEMPORARY TABLE temp_valid_movies AS
        SELECT t.tconst
        FROM titles t
        JOIN title_genres g ON g.tconst = t.tconst
        WHERE {movie_filter}
        GROUP BY t.tconst
        HAVING COUNT(g.genre) > 0
    """)
    conn.execute("CREATE INDEX idx_tvm_tconst ON temp_valid_movies(tconst)")
    
    count_m = conn.execute("SELECT COUNT(*) FROM temp_valid_movies").fetchone()[0]
    print(f"      -> Found {count_m:,} valid movies.")

    print("[3/5] Pre-filtering Valid People...")
    conn.execute("DROP TABLE IF EXISTS temp_valid_people")
    
    # Filter people by birthYear criteria
    conn.execute("""
        CREATE TEMPORARY TABLE temp_valid_people AS
        SELECT nconst
        FROM people p
        WHERE p.birthYear IS NOT NULL 
          AND p.birthYear >= 1800
    """)
    conn.execute("CREATE INDEX idx_tvp_nconst ON temp_valid_people(nconst)")
    
    count_p = conn.execute("SELECT COUNT(*) FROM temp_valid_people").fetchone()[0]
    print(f"      -> Found {count_p:,} valid people.")
    conn.commit()

# --------------------------------------------------------------------------- #
# [4] Bulk Insertion with Progress
# --------------------------------------------------------------------------- #

def batch_insert_edges(conn: sqlite3.Connection, batch_size=10_000):
    print("[4/5] Computing and Inserting Edges...")
    
    # 1. Get list of valid movies to iterate over
    # We iterate over movies (tconst) rather than raw rows to keep batches logical
    cur = conn.cursor()
    cur.execute("SELECT tconst FROM temp_valid_movies")
    all_movies = [r[0] for r in cur.fetchall()]
    
    total_inserted = 0
    
    # 2. Process in chunks
    # This SQL query does the heavy lifting:
    #   - Joins principals ONLY against our small temp tables
    #   - Calculates the hash using the registered Python function
    #   - Inserts directly
    insert_sql = """
        INSERT INTO edges (tconst, nconst, movie_hash, person_hash)
        SELECT 
            pr.tconst, 
            pr.nconst, 
            calc_hash(pr.tconst), 
            calc_hash(pr.nconst)
        FROM principals pr
        -- Filter by our pre-computed valid lists
        INNER JOIN temp_valid_people vp ON vp.nconst = pr.nconst
        WHERE pr.tconst IN ({seq})
        GROUP BY pr.tconst, pr.nconst
    """
    
    # Using tqdm to show progress through the MOVIE list
    pbar = tqdm(total=len(all_movies), unit="movies", desc="Processing")
    
    for i in range(0, len(all_movies), batch_size):
        chunk = all_movies[i : i + batch_size]
        if not chunk:
            break
            
        # Create placeholders for the IN clause
        placeholders = ",".join(["?"] * len(chunk))
        query = insert_sql.format(seq=placeholders)
        
        cur.execute(query, chunk)
        total_inserted += cur.rowcount
        conn.commit()
        
        pbar.update(len(chunk))
        pbar.set_postfix(edges=f"{total_inserted:,}")
        
    pbar.close()
    print(f"      -> Inserted {total_inserted:,} edges.")

# --------------------------------------------------------------------------- #
# [5] Indexing
# --------------------------------------------------------------------------- #

def create_indices(conn: sqlite3.Connection):
    print("[5/5] Building Indices (this may take a moment)...")
    conn.execute("CREATE INDEX idx_edges_tconst ON edges(tconst);")
    conn.execute("CREATE INDEX idx_edges_nconst ON edges(nconst);")
    conn.commit()
    print("      -> Indices built.")

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    # Updated to use the config path
    db_path = Path(project_config.db_path)
    
    print(f"Connecting to training database: {db_path}")
    
    # Connect with optimizations
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA cache_size = -64000") # ~64MB cache
    
    try:
        register_adapters(conn)
        create_edges_table(conn)
        create_valid_filters(conn)
        batch_insert_edges(conn, batch_size=25_000)
        create_indices(conn)
        
        # --- Cleanup Section with Logging ---
        print("[Cleanup] Dropping temporary tables...")
        conn.execute("DROP TABLE IF EXISTS temp_valid_movies")
        conn.execute("DROP TABLE IF EXISTS temp_valid_people")
        
        # Disabled VACUUM to prevent hanging on large DBs (~40min for 17GB).
        # You can run 'sqlite3 data/imdb.db "VACUUM;"' manually if you need to reclaim disk space.
        # print("[Cleanup] Vacuuming database to reclaim disk space (this takes a while)...")
        # conn.execute("VACUUM") 
        
        print("\nAll done. `edges` table ready and optimized.")
        
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    finally:
        conn.close()

if __name__ == "__main__":
    main()