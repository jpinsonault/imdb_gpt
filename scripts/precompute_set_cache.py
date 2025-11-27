# scripts/precompute_set_cache.py

import logging
import sqlite3
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict

from config import ProjectConfig, project_config
from scripts.autoencoder.ae_loader import _load_frozen_autoencoders

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class _RawEntityDataset(Dataset):
    """Helper to serve raw dict rows for batch encoding."""
    def __init__(self, rows, fields):
        self.rows = rows
        self.fields = fields

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        # Return transformed tensors (inputs, targets)
        # Note: fields handle None internally usually, but row must exist
        inputs = [f.transform(row.get(f.name)) for f in self.fields]
        targets = [f.transform_target(row.get(f.name)) for f in self.fields]
        return inputs, targets

def _collate_raw(batch):
    inputs_list, targets_list = zip(*batch)
    # Group inputs by field
    num_fields = len(inputs_list[0])
    collated_inputs = []
    for i in range(num_fields):
        collated_inputs.append(torch.stack([x[i] for x in inputs_list]))
    
    # Group targets by field
    collated_targets = []
    for i in range(num_fields):
        collated_targets.append(torch.stack([x[i] for x in targets_list]))
        
    return collated_inputs, collated_targets

def _fetch_movie_rows(db_path, tconsts, mov_ae) -> List[Dict]:
    """Bulk fetch movie rows to avoid opening 200k sqlite connections."""
    # We'll just reuse the generator logic but filter in python if list is small, 
    # or more efficiently: grab everything and filter. 
    # Given IMDB scale, let's rely on the fact that tconsts are likely contiguous 
    # or we just use the existing row_generator and keep valid ones.
    
    logging.info(f"Bulk fetching {len(tconsts)} movie rows...")
    tconst_set = set(tconsts)
    valid_rows = []
    
    # We use the AE's generator which already does the complex JOINs
    count = 0
    for row in tqdm(mov_ae.row_generator(), desc="Scanning movies"):
        if row['tconst'] in tconst_set:
            valid_rows.append(row)
            count += 1
            
    # Sort to match input order (critical for index alignment)
    row_map = {r['tconst']: r for r in valid_rows}
    ordered_rows = [row_map.get(t, {}) for t in tconsts]
    return ordered_rows

def _fetch_person_rows(db_path, nconsts, per_ae) -> List[Dict]:
    """Bulk fetch person rows."""
    logging.info(f"Bulk fetching {len(nconsts)} person rows...")
    nconst_set = set(nconsts)
    valid_rows = []
    
    for row in tqdm(per_ae.row_generator(), desc="Scanning people"):
        if row['primaryName'] in nconst_set or getattr(row, 'nconst', '') in nconst_set: 
            # Note: row_generator dicts might not have 'nconst' key depending on implementation
            # PeopleAutoencoder generator returns: primaryName, birthYear... 
            # It does NOT return nconst by default in the dict! 
            # We need to patch PeopleAutoencoder.row_generator or rely on order?
            # Actually, let's write a custom quick query here to be safe.
            pass

    # Custom efficient fetch
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Create temp table for join filtering
    cur.execute("CREATE TEMPORARY TABLE target_people (nconst TEXT PRIMARY KEY)")
    cur.executemany("INSERT OR IGNORE INTO target_people (nconst) VALUES (?)", [(n,) for n in nconsts])
    
    # Query mirrors PeopleAutoencoder but joins our temp list
    sql = """
    SELECT 
        p.nconst,
        p.primaryName, 
        p.birthYear, 
        p.deathYear, 
        GROUP_CONCAT(pp.profession, ',')
    FROM people p
    JOIN target_people tp ON tp.nconst = p.nconst
    LEFT JOIN people_professions pp ON pp.nconst = p.nconst
    GROUP BY p.nconst
    """
    
    cur.execute(sql)
    rows = {}
    for r in tqdm(cur, total=len(nconsts), desc="SQL Fetch People"):
        rows[r[0]] = {
            "primaryName": r[1],
            "birthYear": r[2],
            "deathYear": r[3],
            "professions": r[4].split(",") if r[4] else None
        }
    
    ordered_rows = [rows.get(n, {}) for n in nconsts]
    return ordered_rows

def build_set_decoder_cache(cfg: ProjectConfig):
    db_path = Path(cfg.data_dir) / "imdb.db"
    cache_path = Path(cfg.data_dir) / "set_decoder_data.pt"
    
    logging.info("Loading frozen autoencoders...")
    mov_ae, per_ae = _load_frozen_autoencoders(cfg)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Fix for AttributeError: Wrapper class doesn't have .to(), move internals
    mov_ae.encoder.to(device).eval()
    per_ae.encoder.to(device).eval()

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 1. Identify valid Movies and their connected People
    logging.info("Querying edges...")
    # Get all edges: tconst -> list of nconst
    # Limit scope if configured
    limit = getattr(cfg, "set_decoder_movie_limit", None)
    if limit:
        cur.execute("SELECT tconst, nconst FROM edges WHERE tconst IN (SELECT tconst FROM edges GROUP BY tconst LIMIT ?) ORDER BY tconst", (limit,))
    else:
        cur.execute("SELECT tconst, nconst FROM edges ORDER BY tconst")
    
    movie_to_people = {}
    all_nconsts = set()
    
    for t, n in tqdm(cur, desc="Reading edges"):
        if t not in movie_to_people:
            movie_to_people[t] = []
        movie_to_people[t].append(n)
        all_nconsts.add(n)

    all_tconsts = sorted(list(movie_to_people.keys()))
    unique_nconsts = sorted(list(all_nconsts))
    
    logging.info(f"Found {len(all_tconsts)} movies and {len(unique_nconsts)} unique people.")

    # 2. Batch Encode Movies
    # Fetch rows using the AE's generator logic but filtered
    # For speed in this script, we will assume we can fetch by ID or scan efficiently.
    # The custom _fetch helpers above are safest.
    
    # We use a simplified fetch for movies similar to people to ensure alignment
    # Since scanning 10M rows for 10k is slow, we use the temp table trick for movies too.
    
    logging.info("Fetching movie data...")
    cur.execute("CREATE TEMPORARY TABLE target_movies (tconst TEXT PRIMARY KEY)")
    cur.executemany("INSERT OR IGNORE INTO target_movies (tconst) VALUES (?)", [(t,) for t in all_tconsts])
    
    # Mirroring TitlesAutoencoder query
    sql_mov = """
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
    JOIN target_movies tm ON tm.tconst = t.tconst
    JOIN title_genres g ON g.tconst = t.tconst
    GROUP BY t.tconst
    """
    cur.execute(sql_mov)
    
    m_data_map = {}
    for r in tqdm(cur, total=len(all_tconsts), desc="SQL Fetch Movies"):
        m_data_map[r[0]] = {
            "tconst": r[0],
            "primaryTitle": r[1],
            "startYear": r[2],
            "endYear": r[3],
            "runtimeMinutes": r[4],
            "averageRating": r[5],
            "numVotes": r[6],
            "genres": r[7].split(",") if r[7] else []
        }
    
    movie_rows = [m_data_map.get(t, {}) for t in all_tconsts]
    
    mov_ds = _RawEntityDataset(movie_rows, mov_ae.fields)
    mov_dl = DataLoader(mov_ds, batch_size=cfg.batch_size, num_workers=0, collate_fn=_collate_raw)

    mov_latents = []
    with torch.no_grad():
        for inputs, _ in tqdm(mov_dl, desc="Encoding movies"):
            inputs = [x.to(device) for x in inputs]
            z = mov_ae.encoder(inputs) 
            mov_latents.append(z.cpu())

    mov_latents = torch.cat(mov_latents, dim=0)
    
    # 3. Batch Encode People
    logging.info("Fetching people data...")
    person_rows = _fetch_person_rows(db_path, unique_nconsts, per_ae)
    
    per_ds = _RawEntityDataset(person_rows, per_ae.fields)
    per_dl = DataLoader(per_ds, batch_size=cfg.batch_size, num_workers=0, collate_fn=_collate_raw)

    per_latents = []
    per_targets_acc = [[] for _ in per_ae.fields]

    with torch.no_grad():
        for inputs, targets in tqdm(per_dl, desc="Encoding people"):
            inputs = [x.to(device) for x in inputs]
            z = per_ae.encoder(inputs)
            per_latents.append(z.cpu())
            for i, tgt in enumerate(targets):
                per_targets_acc[i].append(tgt.cpu())

    per_latents = torch.cat(per_latents, dim=0)
    per_targets = [torch.cat(acc, dim=0) for acc in per_targets_acc]

    nconst_to_idx = {n: i for i, n in enumerate(unique_nconsts)}

    # 4. Build Index Tensor
    num_slots = int(getattr(cfg, "set_decoder_slots", 10))
    logging.info(f"Building index tensor (slots={num_slots})...")
    
    indices = torch.full((len(all_tconsts), num_slots), -1, dtype=torch.long)
    masks = torch.zeros((len(all_tconsts), num_slots), dtype=torch.bool)
    
    for i, tconst in enumerate(tqdm(all_tconsts, desc="Structuring sets")):
        people = movie_to_people[tconst]
        take = min(len(people), num_slots)
        for k in range(take):
            pidx = nconst_to_idx.get(people[k], -1)
            if pidx != -1:
                indices[i, k] = pidx
                masks[i, k] = True

    # 5. Save
    payload = {
        "movie_latents": mov_latents,
        "person_latents": per_latents,
        "person_targets": per_targets,
        "indices": indices,
        "masks": masks,
        "tconsts": all_tconsts,
    }

    logging.info(f"Saving set decoder cache to {cache_path}...")
    torch.save(payload, cache_path)
    logging.info("Done.")
    return cache_path

def ensure_set_decoder_cache(cfg: ProjectConfig):
    cache_path = Path(cfg.data_dir) / "set_decoder_data.pt"
    if cache_path.exists() and not cfg.refresh_cache:
        logging.info(f"Found existing set decoder cache at {cache_path}")
        return cache_path
    return build_set_decoder_cache(cfg)

if __name__ == "__main__":
    ensure_set_decoder_cache(project_config)