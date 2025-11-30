# scripts/simple_set/precompute.py

import logging
import sqlite3
import torch
import json
from collections import Counter, defaultdict
from pathlib import Path
from tqdm import tqdm
from config import ProjectConfig
from scripts.autoencoder.imdb_row_autoencoders import TitlesAutoencoder
from scripts.sql_filters import movie_select_clause, map_movie_row

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- KNOBS ---
MIN_PERSON_FREQUENCY = 3  # A person must appear in at least this many movies (across all roles)
MIN_PEOPLE_PER_MOVIE = 1  # A movie must have at least this many valid people
# -------------

def _map_category_to_head(category: str) -> str:
    category = (category or "").lower().strip()
    if category in ('actor', 'actress', 'self'):
        return 'cast'
    if category == 'director':
        return 'director'
    if category == 'writer':
        return 'writer'
    return 'crew'

def build_hybrid_cache(cfg: ProjectConfig):
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
    
    # 3. Get valid Role entries
    logging.info("Fetching categorized edges...")
    sql_edges = """
    SELECT p.tconst, p.nconst, p.category
    FROM principals p
    JOIN edges e ON e.tconst = p.tconst AND e.nconst = p.nconst
    """
    try:
        cur.execute(sql_edges)
        raw_rows = cur.fetchall()
    except sqlite3.OperationalError:
        logging.error("Could not query joined edges. Ensure 'edges' and 'principals' tables exist.")
        raise
    
    # 4. Filter Data
    logging.info("Filtering and bucketing...")
    person_counts = Counter(r[1] for r in raw_rows)
    valid_people = {n for n, c in person_counts.items() if c >= MIN_PERSON_FREQUENCY}
    logging.info(f"People filtering: {len(person_counts)} total -> {len(valid_people)} valid")
    
    # Bucket by Movie -> Head -> List[Person]
    movie_data = defaultdict(lambda: defaultdict(set))
    
    for tconst, nconst, category in raw_rows:
        if nconst in valid_people:
            head = _map_category_to_head(category)
            movie_data[tconst][head].add(nconst)
            
    # Identify valid movies
    final_tconsts = []
    for tconst, heads in movie_data.items():
        total_people = sum(len(s) for s in heads.values())
        if total_people >= MIN_PEOPLE_PER_MOVIE:
            final_tconsts.append(tconst)
            
    final_tconsts.sort()
    
    # Build Final Vocab
    final_nconsts_set = set()
    for t in final_tconsts:
        for p_set in movie_data[t].values():
            final_nconsts_set.update(p_set)
            
    sorted_nconsts = sorted(list(final_nconsts_set))
    nconst_to_idx = {n: i for i, n in enumerate(sorted_nconsts)}
    
    num_movies = len(final_tconsts)
    num_people = len(sorted_nconsts)
    
    logging.info(f"Final Vocab: {num_movies} movies, {num_people} people.")
    
    # 5. Precompute Movie Field Tensors
    logging.info("Fetching movie metadata...")
    cur.execute("CREATE TEMPORARY TABLE target_movies (tconst TEXT PRIMARY KEY)")
    cur.executemany("INSERT OR IGNORE INTO target_movies (tconst) VALUES (?)", [(t,) for t in final_tconsts])
    
    sql_meta = f"""
    SELECT 
        {movie_select_clause(alias='t', genre_alias='g')}
    FROM titles t
    JOIN target_movies tm ON tm.tconst = t.tconst
    JOIN title_genres g ON g.tconst = t.tconst
    GROUP BY t.tconst
    """
    cur.execute(sql_meta)
    
    field_data = [[] for _ in mov_ae.fields]
    
    # --- RAGGED TENSOR CONSTRUCTION ---
    # Instead of list-of-dicts, we build 1D flat arrays + offsets for each head
    # This allows O(1) slicing in the dataloader without iterating Python objects
    
    # Temporary storage
    temp_heads = defaultdict(list) # head -> list of (movie_idx, list_of_person_idxs)
    
    fetched_rows = {}
    for r in cur:
        row_dict = map_movie_row(r)
        fetched_rows[row_dict["tconst"]] = row_dict
        
    for idx, tconst in enumerate(tqdm(final_tconsts, desc="Stacking data")):
        row_dict = fetched_rows.get(tconst)
        if not row_dict: continue
            
        heads_data = movie_data[tconst]
        
        # Override peopleCount
        total_p = sum(len(s) for s in heads_data.values())
        row_dict["peopleCount"] = total_p
        
        # 1. Stack Fields
        for i, field in enumerate(mov_ae.fields):
            val = row_dict.get(field.name)
            t = field.transform(val).cpu()
            field_data[i].append(t)
            
        # 2. Collect Ragged Data
        for head_name, nconst_set in heads_data.items():
            idxs = [nconst_to_idx[n] for n in nconst_set if n in nconst_to_idx]
            if idxs:
                # We store just the list of indices. 
                # We will flatten this later.
                temp_heads[head_name].append((idx, idxs))

    # Stack Input Fields
    stacked_fields = []
    for i, field in enumerate(mov_ae.fields):
        stacked = torch.stack(field_data[i])
        stacked_fields.append(stacked)

    # Flatten Targets into Ragged Tensors (Values + Offsets)
    # structure: heads_ragged[head] = {'flat': Tensor, 'offsets': Tensor, 'lengths': Tensor}
    # Offsets array size = num_movies + 1. 
    # offsets[i] -> start index for movie i. 
    # offsets[i+1] -> end index for movie i.
    
    heads_ragged = {}
    
    logging.info("Flattening target indices into tensors...")
    
    # We need to ensure every movie has an entry in the offsets, even if empty.
    # The temp_heads only contains non-empty entries.
    
    unique_heads = list(temp_heads.keys())
    
    for head in unique_heads:
        # 1. Create lists for all movies (dense list of lists)
        # Using a list comprehension is faster than dict lookups in a loop
        # But we need to map back to the movie index `idx` from the loop above.
        
        # Re-organize temp_heads into a dense lookup
        # movie_idx -> [p1, p2...]
        dense_map = {}
        for m_idx, p_list in temp_heads[head]:
            dense_map[m_idx] = p_list
            
        flat_values = []
        lengths = []
        
        for i in range(num_movies):
            p_list = dense_map.get(i, [])
            flat_values.extend(p_list)
            lengths.append(len(p_list))
            
        # Convert to tensors
        flat_tensor = torch.tensor(flat_values, dtype=torch.long)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)
        
        # Compute offsets (cumulative sum)
        # offsets[0] = 0, offsets[1] = len(movie_0), etc.
        offsets_tensor = torch.zeros(num_movies + 1, dtype=torch.long)
        offsets_tensor[1:] = torch.cumsum(lengths_tensor, dim=0)
        
        heads_ragged[head] = {
            "flat": flat_tensor,
            "offsets": offsets_tensor,
            "lengths": lengths_tensor # Useful for counts
        }

    # 6. Fetch Names
    logging.info("Fetching person names...")
    idx_to_person_name = {}
    cur.execute("CREATE TEMPORARY TABLE temp_people (nconst TEXT PRIMARY KEY)")
    cur.executemany("INSERT OR IGNORE INTO temp_people (nconst) VALUES (?)", [(n,) for n in sorted_nconsts])
    cur.execute("SELECT p.nconst, p.primaryName FROM people p JOIN temp_people tp ON tp.nconst = p.nconst")
    
    for n, name in cur:
        if n in nconst_to_idx:
            idx_to_person_name[nconst_to_idx[n]] = name
            
    conn.close()
    
    # 7. Field Stats
    field_configs = {}
    for f in mov_ae.fields:
        from scripts.autoencoder.row_autoencoder import _field_to_state
        field_configs[f.name] = _field_to_state(f)

    payload = {
        "num_people": num_people,
        "stacked_fields": stacked_fields,   
        "heads_ragged": heads_ragged,       # Optimized sparse storage
        "idx_to_person_name": idx_to_person_name,
        "field_configs": field_configs,
        "field_names": [f.name for f in mov_ae.fields]
    }
    
    logging.info(f"Saving hybrid cache to {cache_path}...")
    torch.save(payload, cache_path)
    return cache_path

def ensure_hybrid_cache(cfg: ProjectConfig):
    cache_path = Path(cfg.data_dir) / "hybrid_set_cache.pt"
    if cache_path.exists() and not cfg.refresh_cache:
        logging.info(f"Found existing hybrid cache at {cache_path}")
        return cache_path
    return build_hybrid_cache(cfg)