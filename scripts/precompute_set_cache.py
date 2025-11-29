# scripts/autoencoder/precompute_set_cache.py

import logging
import sqlite3
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict

from config import ProjectConfig, project_config
from scripts.autoencoder.ae_loader import _load_frozen_autoencoders

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class _RawEntityDataset(Dataset):
    def __init__(self, rows, fields):
        self.rows = rows
        self.fields = fields

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        inputs = [f.transform(row.get(f.name)) for f in self.fields]
        targets = [f.transform_target(row.get(f.name)) for f in self.fields]
        return inputs, targets

def _collate_raw(batch):
    inputs_list, targets_list = zip(*batch)
    num_fields = len(inputs_list[0])
    collated_inputs = []
    for i in range(num_fields):
        collated_inputs.append(torch.stack([x[i] for x in inputs_list]))
    
    collated_targets = []
    for i in range(num_fields):
        collated_targets.append(torch.stack([x[i] for x in targets_list]))
        
    return collated_inputs, collated_targets

def _fetch_person_rows(db_path, nconsts, per_ae) -> List[Dict]:
    logging.info(f"Bulk fetching {len(nconsts)} person rows...")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    cur.execute("CREATE TEMPORARY TABLE target_people (nconst TEXT PRIMARY KEY)")
    cur.executemany("INSERT OR IGNORE INTO target_people (nconst) VALUES (?)", [(n,) for n in nconsts])
    
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

def build_sequence_cache(cfg: ProjectConfig):
    db_path = Path(cfg.data_dir) / "imdb.db"
    cache_path = Path(cfg.data_dir) / "seq_decoder_data.pt"
    
    logging.info("Loading autoencoders (for field defs only)...")
    mov_ae, per_ae = _load_frozen_autoencoders(cfg)
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 1. Identify valid Movies and their ordered People
    logging.info("Querying ordered principals...")
    limit_clause = ""
    limit = getattr(cfg, "movie_limit", None)
    if limit and limit < 100000000000:
        limit_clause = f"LIMIT {limit}"

    cur.execute(f"SELECT tconst FROM edges GROUP BY tconst {limit_clause}")
    valid_tconsts = {r[0] for r in cur.fetchall()}

    logging.info(f"Fetching ordered cast for {len(valid_tconsts)} movies...")
    
    cur.execute(f"""
        SELECT pr.tconst, pr.nconst 
        FROM principals pr
        WHERE pr.tconst IN (SELECT tconst FROM edges)
        ORDER BY pr.tconst, pr.ordering
    """)

    movie_to_people = {}
    all_nconsts = set()
    
    for t, n in tqdm(cur, desc="Organizing sequences"):
        if t not in valid_tconsts: continue
        if t not in movie_to_people:
            movie_to_people[t] = []
        movie_to_people[t].append(n)
        all_nconsts.add(n)

    all_tconsts = sorted([t for t in movie_to_people.keys() if t in valid_tconsts])
    unique_nconsts = sorted(list(all_nconsts))
    
    logging.info(f"Found {len(all_tconsts)} valid movies and {len(unique_nconsts)} unique people.")

    # 2. Batch Process Movies (Store Raw Inputs)
    logging.info("Fetching movie data...")
    cur.execute("CREATE TEMPORARY TABLE target_movies (tconst TEXT PRIMARY KEY)")
    cur.executemany("INSERT OR IGNORE INTO target_movies (tconst) VALUES (?)", [(t,) for t in all_tconsts])
    
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

    mov_inputs_acc = [[] for _ in mov_ae.fields]
    
    # We store the *inputs* to the encoder, not the latents
    for inputs, _ in tqdm(mov_dl, desc="Processing movie inputs"):
        for i, tensor in enumerate(inputs):
            mov_inputs_acc[i].append(tensor.cpu())

    mov_inputs = [torch.cat(acc, dim=0) for acc in mov_inputs_acc]

    # 3. Batch Process People (Store Raw Inputs & Targets)
    logging.info("Fetching people data...")
    person_rows = _fetch_person_rows(db_path, unique_nconsts, per_ae)
    
    per_ds = _RawEntityDataset(person_rows, per_ae.fields)
    per_dl = DataLoader(per_ds, batch_size=cfg.batch_size, num_workers=0, collate_fn=_collate_raw)

    per_inputs_acc = [[] for _ in per_ae.fields]
    per_targets_acc = [[] for _ in per_ae.fields]

    for inputs, targets in tqdm(per_dl, desc="Processing people inputs"):
        for i, tensor in enumerate(inputs):
            per_inputs_acc[i].append(tensor.cpu())
        for i, tensor in enumerate(targets):
            per_targets_acc[i].append(tensor.cpu())

    per_inputs = [torch.cat(acc, dim=0) for acc in per_inputs_acc]
    per_targets = [torch.cat(acc, dim=0) for acc in per_targets_acc]

    nconst_to_idx = {n: i for i, n in enumerate(unique_nconsts)}

    # 4. Build Index Tensor
    seq_len = int(getattr(cfg, "seq_decoder_len", 10))
    logging.info(f"Building sequence tensor (max_len={seq_len})...")
    
    indices = torch.full((len(all_tconsts), seq_len), -1, dtype=torch.long)
    masks = torch.zeros((len(all_tconsts), seq_len), dtype=torch.bool)
    
    for i, tconst in enumerate(tqdm(all_tconsts, desc="Building sequences")):
        people = movie_to_people[tconst]
        take = min(len(people), seq_len)
        for k in range(take):
            pidx = nconst_to_idx.get(people[k], -1)
            if pidx != -1:
                indices[i, k] = pidx
                masks[i, k] = True

    # 5. Save
    payload = {
        # Store lists of tensors (fields)
        "movie_inputs": mov_inputs,
        "person_inputs": per_inputs,
        "person_targets": per_targets,
        "indices": indices,
        "masks": masks,
        "tconsts": all_tconsts,
    }

    logging.info(f"Saving sequence input cache to {cache_path}...")
    torch.save(payload, cache_path)
    logging.info("Done.")
    return cache_path

def ensure_set_decoder_cache(cfg: ProjectConfig):
    cache_path = Path(cfg.data_dir) / "seq_decoder_data.pt"
    if cache_path.exists() and not cfg.refresh_cache:
        logging.info(f"Found existing sequence cache at {cache_path}")
        return cache_path
    return build_sequence_cache(cfg)

if __name__ == "__main__":
    ensure_set_decoder_cache(project_config)