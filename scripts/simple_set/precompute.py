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
MIN_PERSON_FREQUENCY = 3  # A person must appear in at least this many movies
PADDING_IDX = -1          # Value used to pad the dense tensors
# -------------

def _map_category_to_head(category: str) -> str | None:
    """
    Maps raw categories to specific heads. 
    Returns None for categories we don't track (like 'producer', 'editor', 'crew').
    """
    category = (category or "").lower().strip()
    if category in ('actor', 'actress', 'self'):
        return 'cast'
    if category == 'director':
        return 'director'
    if category == 'writer':
        return 'writer'
    return None

def build_hybrid_cache(cfg: ProjectConfig):
    db_path = Path(cfg.db_path)
    cache_path = Path(cfg.data_dir) / "hybrid_set_cache.pt"
    
    # 1. Setup Movie Fields
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
    
    # Track which people appear in which head to build subsets
    # head -> set(nconst)
    head_populations = defaultdict(set)

    for tconst, nconst, category in raw_rows:
        if nconst in valid_people:
            head = _map_category_to_head(category)
            if head is not None:
                movie_data[tconst][head].add(nconst)
                head_populations[head].add(nconst)
            
    # Identify valid movies
    final_tconsts = []
    skipped_count = 0
    
    for tconst, heads in movie_data.items():
        # STRICT FILTER: Must have Cast AND Director
        has_cast = len(heads.get('cast', [])) > 0
        has_director = len(heads.get('director', [])) > 0
        
        if has_cast and has_director:
            final_tconsts.append(tconst)
        else:
            skipped_count += 1
            
    final_tconsts.sort()
    logging.info(f"Movie filtering: Kept {len(final_tconsts)} valid movies (Skipped {skipped_count} missing cast/director).")
    
    # Build Final Global Vocab (Recalculate based on filtered movies)
    final_nconsts_set = set()
    for t in final_tconsts:
        for p_set in movie_data[t].values():
            final_nconsts_set.update(p_set)
            
    sorted_nconsts = sorted(list(final_nconsts_set))
    nconst_to_global_idx = {n: i for i, n in enumerate(sorted_nconsts)}
    
    num_movies = len(final_tconsts)
    num_people = len(sorted_nconsts)
    
    logging.info(f"Final Vocab: {num_movies} movies, {num_people} global people.")
    
    # --- Build Head Subsets & Mappings ---
    # We create a mapping tensor for each head: Mapping[Global_ID] -> Local_ID
    # If Global_ID is not in that head, value is -1.
    head_mappings = {} # head_name -> Tensor(num_people)
    head_vocab_sizes = {} # head_name -> int

    logging.info("Building head-specific subsets and mappings...")
    for head, nconsts in head_populations.items():
        # Only include nconsts that survived the global filter
        valid_head_nconsts = [n for n in nconsts if n in nconst_to_global_idx]
        valid_head_nconsts.sort() # Ensure deterministic order
        
        local_vocab_size = len(valid_head_nconsts)
        head_vocab_sizes[head] = local_vocab_size
        
        # Create Mapping Tensor: Default -1
        # Size = Global Vocab Size
        mapping_tensor = torch.full((num_people,), -1, dtype=torch.long)
        
        for local_idx, nconst in enumerate(valid_head_nconsts):
            global_idx = nconst_to_global_idx[nconst]
            mapping_tensor[global_idx] = local_idx
            
        head_mappings[head] = mapping_tensor
        logging.info(f"  Head '{head}': {local_vocab_size} people ({local_vocab_size/num_people:.1%} of global).")

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
    
    # Temporary storage: head -> list of lists (one list of GLOBAL idxs per movie)
    temp_heads_lists = defaultdict(lambda: [[] for _ in range(num_movies)])
    
    fetched_rows = {}
    for r in cur:
        row_dict = map_movie_row(r)
        fetched_rows[row_dict["tconst"]] = row_dict
        
    for idx, tconst in enumerate(tqdm(final_tconsts, desc="Stacking data")):
        row_dict = fetched_rows.get(tconst)
        if not row_dict: continue
            
        heads_data = movie_data[tconst]
        
        # Override peopleCount (recalculate based on filtered heads)
        total_p = sum(len(s) for s in heads_data.values())
        row_dict["peopleCount"] = total_p
        
        # 1. Stack Fields
        for i, field in enumerate(mov_ae.fields):
            val = row_dict.get(field.name)
            t = field.transform(val).cpu()
            field_data[i].append(t)
            
        # 2. Collect Indices per Head (Store GLOBAL indices here)
        for head_name, nconst_set in heads_data.items():
            idxs = [nconst_to_global_idx[n] for n in nconst_set if n in nconst_to_global_idx]
            if idxs:
                temp_heads_lists[head_name][idx] = idxs

    # Stack Input Fields
    stacked_fields = []
    for i, field in enumerate(mov_ae.fields):
        stacked = torch.stack(field_data[i])
        stacked_fields.append(stacked)

    # Build Padded Tensors (Using GLOBAL indices)
    # We store global indices in the dataset. We will map them to local indices during training.
    heads_padded = {}
    logging.info("Building padded target tensors...")
    
    for head, lists in temp_heads_lists.items():
        max_len = max(len(l) for l in lists) if lists else 0
        max_len = max(1, max_len) 
        
        padded_tensor = torch.full((num_movies, max_len), PADDING_IDX, dtype=torch.int32)
        
        for i, idx_list in enumerate(lists):
            if idx_list:
                length = len(idx_list)
                padded_tensor[i, :length] = torch.tensor(idx_list, dtype=torch.int32)
                
        heads_padded[head] = padded_tensor

    # 6. Fetch Names
    logging.info("Fetching person names...")
    idx_to_person_name = {}
    cur.execute("CREATE TEMPORARY TABLE temp_people (nconst TEXT PRIMARY KEY)")
    cur.executemany("INSERT OR IGNORE INTO temp_people (nconst) VALUES (?)", [(n,) for n in sorted_nconsts])
    cur.execute("SELECT p.nconst, p.primaryName FROM people p JOIN temp_people tp ON tp.nconst = p.nconst")
    
    for n, name in cur:
        if n in nconst_to_global_idx:
            idx_to_person_name[nconst_to_global_idx[n]] = name
            
    conn.close()
    
    # 7. Field Stats
    field_configs = {}
    for f in mov_ae.fields:
        from scripts.autoencoder.row_autoencoder import _field_to_state
        field_configs[f.name] = _field_to_state(f)

    payload = {
        "num_people": num_people, # Global count
        "stacked_fields": stacked_fields,   
        "heads_padded": heads_padded,       # Stores Global Indices
        "head_mappings": head_mappings,     # New: Tensor[Global] -> Local
        "head_vocab_sizes": head_vocab_sizes, # New: Local sizes
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