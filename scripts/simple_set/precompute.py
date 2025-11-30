# scripts/simple_set/precompute.py

import logging
import sqlite3
import torch
import json
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from config import ProjectConfig
from scripts.autoencoder.imdb_row_autoencoders import TitlesAutoencoder
from scripts.sql_filters import movie_select_clause, map_movie_row

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- KNOBS ---
MIN_PERSON_FREQUENCY = 3  # A person must appear in at least this many movies
MIN_PEOPLE_PER_MOVIE = 1  # A movie must have at least this many valid people
# -------------

def build_hybrid_cache(cfg: ProjectConfig):
    # Updated to use cfg.db_path
    db_path = Path(cfg.db_path)
    cache_path = Path(cfg.data_dir) / "hybrid_set_cache.pt"
    
    # 1. Setup Movie Fields using the Autoencoder class
    logging.info("Initializing TitlesAutoencoder to learn field stats...")
    mov_ae = TitlesAutoencoder(cfg)
    mov_ae.accumulate_stats()
    mov_ae.finalize_stats()
    
    # 2. Connect DB
    logging.info(f"Connecting to {db_path}...")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # 3. Get valid edges (Movie <-> Person)
    logging.info("Fetching edges...")
    try:
        cur.execute("SELECT tconst, nconst FROM edges")
        raw_edges = cur.fetchall()
    except sqlite3.OperationalError:
        logging.error("Could not find 'edges' table. Did you run 'scripts/precompute_edges_table.py'?")
        raise
    
    # 4. Filter Data (The Knobs)
    logging.info("Filtering edges...")
    
    # A. Count appearances of each person
    person_counts = Counter(n for t, n in raw_edges)
    
    # B. Identify valid people (Knob: MIN_PERSON_FREQUENCY)
    valid_people = {n for n, c in person_counts.items() if c >= MIN_PERSON_FREQUENCY}
    logging.info(f"People filtering: {len(person_counts)} total -> {len(valid_people)} valid (>= {MIN_PERSON_FREQUENCY} appearances)")
    
    # C. Group by movie, excluding invalid people
    movie_to_people = {}
    for t, n in raw_edges:
        if n in valid_people:
            if t not in movie_to_people:
                movie_to_people[t] = []
            movie_to_people[t].append(n)
            
    # D. Identify valid movies (Knob: MIN_PEOPLE_PER_MOVIE)
    final_movie_to_people = {
        t: p_list for t, p_list in movie_to_people.items() 
        if len(p_list) >= MIN_PEOPLE_PER_MOVIE
    }
    
    all_tconsts = sorted(list(final_movie_to_people.keys()))
    
    # Re-calculate final person vocabulary based ONLY on surviving movies
    final_nconsts_set = set()
    for p_list in final_movie_to_people.values():
        final_nconsts_set.update(p_list)
        
    sorted_nconsts = sorted(list(final_nconsts_set))
    nconst_to_idx = {n: i for i, n in enumerate(sorted_nconsts)}
    
    num_movies = len(all_tconsts)
    num_people = len(sorted_nconsts)
    
    logging.info(f"Final Vocab: {num_movies} movies, {num_people} people.")
    logging.info(f"Dropped {len(person_counts) - num_people} people and {len(set(t for t,n in raw_edges)) - num_movies} movies during filtering.")
    
    # 5. Precompute Movie Field Tensors
    logging.info("Fetching movie metadata...")
    cur.execute("CREATE TEMPORARY TABLE target_movies (tconst TEXT PRIMARY KEY)")
    cur.executemany("INSERT OR IGNORE INTO target_movies (tconst) VALUES (?)", [(t,) for t in all_tconsts])
    
    sql = f"""
    SELECT 
        {movie_select_clause(alias='t', genre_alias='g')}
    FROM titles t
    JOIN target_movies tm ON tm.tconst = t.tconst
    JOIN title_genres g ON g.tconst = t.tconst
    GROUP BY t.tconst
    """
    
    cur.execute(sql)
    
    # Store transformed tensors in lists
    # Structure: field_data[field_index] = list of tensors
    field_data = [[] for _ in mov_ae.fields]
    
    # Target indices for people
    target_indices_list = []
    
    logging.info("Transforming movie rows...")
    
    # Fetch all relevant rows first to ensure we process them in the sorted order of `all_tconsts`
    fetched_rows = {}
    for r in cur:
        row_dict = map_movie_row(r)
        fetched_rows[row_dict["tconst"]] = row_dict

    valid_count = 0
    missing_count = 0
    
    for tconst in tqdm(all_tconsts, desc="Stacking data"):
        if tconst not in fetched_rows:
            missing_count += 1
            continue
            
        row_dict = fetched_rows[tconst]
        
        # --- CRITICAL FIX ---
        # Override the SQL-derived 'peopleCount' (which is raw from DB) 
        # with the ACTUAL filtered count for this specific dataset.
        people = final_movie_to_people[tconst]
        row_dict["peopleCount"] = len(people)
        # --------------------
        
        # Transform fields
        for i, field in enumerate(mov_ae.fields):
            val = row_dict.get(field.name)
            t = field.transform(val).cpu()
            field_data[i].append(t)
            
        # Get targets
        p_idxs = [nconst_to_idx[n] for n in people]
        target_indices_list.append(p_idxs)
        valid_count += 1

    if missing_count > 0:
        logging.warning(f"{missing_count} movies were in edge list but missing metadata in titles table.")

    # Convert lists of tensors to stacked tensors
    logging.info("Stacking tensors...")
    stacked_fields = []
    for i, field in enumerate(mov_ae.fields):
        # Stack: (NumMovies, FeatureDim...)
        stacked = torch.stack(field_data[i])
        stacked_fields.append(stacked)

    # 6. Fetch Person Names for Recon
    logging.info("Fetching person names...")
    idx_to_person_name = {}
    
    cur.execute("CREATE TEMPORARY TABLE temp_people (nconst TEXT PRIMARY KEY)")
    cur.executemany("INSERT OR IGNORE INTO temp_people (nconst) VALUES (?)", [(n,) for n in sorted_nconsts])
    cur.execute("SELECT p.nconst, p.primaryName FROM people p JOIN temp_people tp ON tp.nconst = p.nconst")
    
    for n, name in cur:
        if n in nconst_to_idx:
            idx_to_person_name[nconst_to_idx[n]] = name
            
    conn.close()
    
    # 7. Save Field Config state
    field_configs = {}
    for f in mov_ae.fields:
        from scripts.autoencoder.row_autoencoder import _field_to_state
        field_configs[f.name] = _field_to_state(f)

    payload = {
        "num_people": num_people,
        "stacked_fields": stacked_fields,   # List[Tensor]
        "target_indices": target_indices_list, # List[List[int]] (ragged)
        "idx_to_person_name": idx_to_person_name,
        "field_configs": field_configs,     # Dict[str, dict]
        "field_names": [f.name for f in mov_ae.fields]
    }
    
    logging.info(f"Saving hybrid cache to {cache_path}...")
    torch.save(payload, cache_path)
    logging.info("Done.")
    return cache_path

def ensure_hybrid_cache(cfg: ProjectConfig):
    cache_path = Path(cfg.data_dir) / "hybrid_set_cache.pt"
    if cache_path.exists() and not cfg.refresh_cache:
        logging.info(f"Found existing hybrid cache at {cache_path}")
        return cache_path
    return build_hybrid_cache(cfg)