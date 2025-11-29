# scripts/simple_set/precompute.py

import logging
import sqlite3
import torch
import json
from pathlib import Path
from tqdm import tqdm
from config import ProjectConfig
from scripts.autoencoder.imdb_row_autoencoders import TitlesAutoencoder
from scripts.sql_filters import movie_select_clause, map_movie_row

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def build_hybrid_cache(cfg: ProjectConfig):
    db_path = Path(cfg.data_dir) / "imdb.db"
    cache_path = Path(cfg.data_dir) / "hybrid_set_cache.pt"
    
    # 1. Setup Movie Fields using the Autoencoder class
    # We use this class to manage stats, tokenizers, and transformations
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
    cur.execute("SELECT tconst, nconst FROM edges")
    raw_edges = cur.fetchall()
    
    # 4. Map Data
    movie_to_people = {}
    all_nconsts = set()
    all_tconsts = set()
    
    for t, n in tqdm(raw_edges, desc="Grouping edges"):
        if t not in movie_to_people:
            movie_to_people[t] = []
        movie_to_people[t].append(n)
        all_nconsts.add(n)
        all_tconsts.add(t)
        
    sorted_nconsts = sorted(list(all_nconsts))
    sorted_tconsts = sorted(list(all_tconsts)) # Ensure deterministic order
    
    nconst_to_idx = {n: i for i, n in enumerate(sorted_nconsts)}
    num_people = len(sorted_nconsts)
    
    logging.info(f"Vocab: {len(sorted_tconsts)} movies, {num_people} people.")
    
    # 5. Precompute Movie Field Tensors
    # We cannot store raw text efficiently, so we store the tensor outputs of field.transform()
    
    # Bulk fetch movie data
    logging.info("Fetching movie metadata...")
    cur.execute("CREATE TEMPORARY TABLE target_movies (tconst TEXT PRIMARY KEY)")
    cur.executemany("INSERT OR IGNORE INTO target_movies (tconst) VALUES (?)", [(t,) for t in sorted_tconsts])
    
    # Use centralized SQL clause
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
    
    # Metadata for Recon
    idx_to_person_name = {} # Load later
    
    logging.info("Transforming movie rows...")
    
    valid_count = 0
    
    for r in tqdm(cur, total=len(sorted_tconsts), desc="Processing movies"):
        row_dict = map_movie_row(r)
        tconst = row_dict["tconst"]
        
        # Transform fields
        for i, field in enumerate(mov_ae.fields):
            val = row_dict.get(field.name)
            # Transform returns a tensor. Clone to CPU to be safe.
            t = field.transform(val).cpu()
            field_data[i].append(t)
            
        # Get targets
        people = movie_to_people.get(tconst, [])
        p_idxs = [nconst_to_idx[n] for n in people]
        target_indices_list.append(p_idxs)
        valid_count += 1

    # Convert lists of tensors to stacked tensors
    logging.info("Stacking tensors...")
    stacked_fields = []
    for i, field in enumerate(mov_ae.fields):
        # Stack: (NumMovies, FeatureDim...)
        stacked = torch.stack(field_data[i])
        stacked_fields.append(stacked)

    # 6. Fetch Person Names for Recon
    logging.info("Fetching person names...")
    cur.execute("CREATE TEMPORARY TABLE temp_people (nconst TEXT PRIMARY KEY)")
    cur.executemany("INSERT OR IGNORE INTO temp_people (nconst) VALUES (?)", [(n,) for n in sorted_nconsts])
    cur.execute("SELECT p.nconst, p.primaryName FROM people p JOIN temp_people tp ON tp.nconst = p.nconst")
    
    for n, name in cur:
        if n in nconst_to_idx:
            idx_to_person_name[nconst_to_idx[n]] = name
            
    conn.close()
    
    # 7. Save Field Config state so we can rebuild the model correctly
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