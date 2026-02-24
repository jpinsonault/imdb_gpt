# scripts/simple_set/precompute.py

import logging
import sqlite3
import torch
from collections import Counter, defaultdict
from pathlib import Path
from tqdm import tqdm

from config import ProjectConfig
from scripts.autoencoder.imdb_row_autoencoders import TitlesAutoencoder, PeopleAutoencoder
from scripts.sql_filters import (
    movie_select_clause,
    map_movie_row,
    people_select_clause,
    map_person_row,
)
from scripts.autoencoder.row_autoencoder import _field_to_state

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PADDING_IDX = -1            # Value used to pad the dense tensors


def _map_category_to_head(category: str, cfg: ProjectConfig) -> str | None:
    return cfg.hybrid_set_category_to_head.get((category or "").lower().strip())


# ---------------------------------------------------------------------------
# Phase 1: Initialize autoencoders and accumulate field stats
# ---------------------------------------------------------------------------

def _init_autoencoders(cfg):
    """Build TitlesAutoencoder and PeopleAutoencoder, accumulate and finalize field stats."""
    logging.info("Initializing autoencoders and learning field stats...")
    mov_ae = TitlesAutoencoder(cfg)
    people_ae = PeopleAutoencoder(cfg)

    force_no_cache = not cfg.use_cache or cfg.refresh_cache
    if force_no_cache:
        logging.info("Temporarily forcing non-cached stats accumulation...")
        mov_ae._drop_cache()
        people_ae._drop_cache()

    mov_ae.accumulate_stats()
    mov_ae.finalize_stats()
    people_ae.accumulate_stats()
    people_ae.finalize_stats()

    if force_no_cache:
        cfg.refresh_cache = False

    return mov_ae, people_ae


# ---------------------------------------------------------------------------
# Phase 2: Fetch edges from DB and bucket by head type
# ---------------------------------------------------------------------------

def _fetch_and_bucket_edges(cur, cfg):
    """
    Load all principal-edge rows, filter by person frequency, and bucket
    into movie_data[tconst][head] = {nconsts} and person_data[nconst][head] = {tconsts}.

    Returns (movie_data, person_data, movie_head_populations, person_head_populations).
    """
    logging.info("Fetching categorized edges...")
    cur.execute("""
        SELECT p.tconst, p.nconst, p.category
        FROM principals p
        JOIN edges e ON e.tconst = p.tconst AND e.nconst = p.nconst
    """)
    raw_rows = cur.fetchall()

    logging.info("Filtering by person frequency and bucketing by head...")
    person_counts = Counter(r[1] for r in raw_rows)
    valid_people = {n for n, c in person_counts.items() if c >= cfg.hybrid_set_min_person_frequency}

    movie_data = defaultdict(lambda: defaultdict(set))
    person_data = defaultdict(lambda: defaultdict(set))
    movie_head_populations = defaultdict(set)
    person_head_populations = defaultdict(set)

    for tconst, nconst, category in raw_rows:
        if nconst not in valid_people:
            continue
        head = _map_category_to_head(category, cfg)
        if head is None:
            continue
        movie_data[tconst][head].add(nconst)
        person_data[nconst][head].add(tconst)
        movie_head_populations[head].add(nconst)
        person_head_populations[head].add(tconst)

    return movie_data, person_data, movie_head_populations, person_head_populations


# ---------------------------------------------------------------------------
# Phase 3: Filter movies and build global index mappings
# ---------------------------------------------------------------------------

def _build_global_indices(cur, movie_data, potential_tconsts):
    """
    Fetch movie metadata, intersect with potential_tconsts, and build
    tconst-to-index and nconst-to-index mappings.

    Returns (final_tconsts, fetched_rows, tconst_to_global_idx,
             sorted_nconsts, nconst_to_global_idx).
    """
    logging.info("Fetching movie metadata and building global indices...")
    cur.execute("CREATE TEMPORARY TABLE target_movies (tconst TEXT PRIMARY KEY)")
    cur.executemany(
        "INSERT OR IGNORE INTO target_movies (tconst) VALUES (?)",
        [(t,) for t in potential_tconsts],
    )

    sql_meta = f"""
    SELECT
        {movie_select_clause(alias='t', genre_alias='g')}
    FROM titles t
    JOIN target_movies tm ON tm.tconst = t.tconst
    JOIN title_genres g ON g.tconst = t.tconst
    GROUP BY t.tconst
    """
    cur.execute(sql_meta)

    fetched_rows = {}
    for r in cur:
        row_dict = map_movie_row(r)
        fetched_rows[row_dict["tconst"]] = row_dict

    final_tconsts = [t for t in potential_tconsts if t in fetched_rows]
    logging.info(
        f"Metadata alignment: {len(final_tconsts)} movies confirmed "
        f"(dropped {len(potential_tconsts) - len(final_tconsts)} missing metadata)."
    )

    tconst_to_global_idx = {t: i for i, t in enumerate(final_tconsts)}

    # Collect all people referenced by the final movies
    final_nconsts_set = set()
    for t in final_tconsts:
        for p_set in movie_data[t].values():
            final_nconsts_set.update(p_set)

    sorted_nconsts = sorted(final_nconsts_set)
    nconst_to_global_idx = {n: i for i, n in enumerate(sorted_nconsts)}

    return final_tconsts, fetched_rows, tconst_to_global_idx, sorted_nconsts, nconst_to_global_idx


# ---------------------------------------------------------------------------
# Phase 4: Build head vocabulary mappings (local ↔ global index translation)
# ---------------------------------------------------------------------------

def _build_head_vocab_mappings(head_populations, id_to_global_idx, total_items):
    """
    For each head, build:
      - vocab_sizes[head]: number of items active in this head
      - mappings[head]: tensor mapping global_idx → local_idx (or -1)
      - local_to_global[head]: tensor mapping local_idx → global_idx
    """
    vocab_sizes = {}
    mappings = {}
    local_to_global = {}

    for head, item_ids in head_populations.items():
        valid_ids = sorted(i for i in item_ids if i in id_to_global_idx)
        local_vocab_size = len(valid_ids)
        vocab_sizes[head] = local_vocab_size

        mapping_tensor = torch.full((total_items,), -1, dtype=torch.long)
        for local_idx, item_id in enumerate(valid_ids):
            mapping_tensor[id_to_global_idx[item_id]] = local_idx
        mappings[head] = mapping_tensor

        if local_vocab_size > 0:
            l2g = torch.full((local_vocab_size,), -1, dtype=torch.long)
            valid_mask = mapping_tensor != -1
            global_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
            local_indices = mapping_tensor[global_indices]
            l2g[local_indices] = global_indices
            local_to_global[head] = l2g

    return vocab_sizes, mappings, local_to_global


# ---------------------------------------------------------------------------
# Phase 5: Stack field tensors and pad head target lists
# ---------------------------------------------------------------------------

def _stack_entity_fields(entity_ids, id_to_entity_data, head_data_lookup,
                         ae_fields, cfg, id_to_global_idx, desc):
    """
    For each entity (movie or person), transform field values into tensors
    and collect head target index lists.

    Returns (stacked_fields, heads_padded, temp_heads_lists).
    """
    num_entities = len(entity_ids)
    field_data = [[] for _ in ae_fields]
    temp_heads_lists = defaultdict(lambda: [[] for _ in range(num_entities)])

    for idx, entity_id in enumerate(tqdm(entity_ids, desc=desc)):
        row_dict = id_to_entity_data.get(entity_id, {})
        if not isinstance(row_dict, dict):
            row_dict = dict(row_dict)

        heads_data = head_data_lookup(entity_id)
        for head_name in cfg.hybrid_set_heads:
            row_dict[f"{head_name}Count"] = len(heads_data.get(head_name, []))

        for i, f in enumerate(ae_fields):
            val = row_dict.get(f.name)
            t = f.transform(val).cpu()
            field_data[i].append(t)

        for head_name, target_ids in heads_data.items():
            idxs = [id_to_global_idx[t] for t in target_ids if t in id_to_global_idx]
            if idxs:
                temp_heads_lists[head_name][idx] = idxs

    stacked_fields = [torch.stack(field_data[i]) for i in range(len(ae_fields))]
    heads_padded = _pad_heads_lists(temp_heads_lists, num_entities)

    return stacked_fields, heads_padded, temp_heads_lists


def _pad_heads_lists(temp_heads_lists, num_entities):
    """Convert ragged lists-of-lists into padded tensors."""
    heads_padded = {}
    for head, lists in temp_heads_lists.items():
        max_len = max(1, max(len(l) for l in lists) if lists else 0)
        padded = torch.full((num_entities, max_len), PADDING_IDX, dtype=torch.int32)
        for i, idx_list in enumerate(lists):
            if idx_list:
                padded[i, :len(idx_list)] = torch.tensor(idx_list, dtype=torch.int32)
        heads_padded[head] = padded
    return heads_padded


# ---------------------------------------------------------------------------
# Phase 6: Fetch person names and metadata
# ---------------------------------------------------------------------------

def _fetch_person_names(cur, sorted_nconsts, nconst_to_global_idx):
    """Fetch primaryName for all people and return {global_idx: name}."""
    logging.info("Fetching person names...")
    cur.execute("CREATE TEMPORARY TABLE temp_people (nconst TEXT PRIMARY KEY)")
    cur.executemany(
        "INSERT OR IGNORE INTO temp_people (nconst) VALUES (?)",
        [(n,) for n in sorted_nconsts],
    )
    cur.execute(
        "SELECT p.nconst, p.primaryName FROM people p "
        "JOIN temp_people tp ON tp.nconst = p.nconst"
    )
    idx_to_name = {}
    for n, name in cur:
        if n in nconst_to_global_idx:
            idx_to_name[nconst_to_global_idx[n]] = name
    return idx_to_name


def _fetch_person_metadata(cur, sorted_nconsts):
    """Fetch full person metadata rows keyed by nconst."""
    logging.info("Fetching person metadata...")
    cur.execute("CREATE TEMPORARY TABLE target_people (nconst TEXT PRIMARY KEY)")
    cur.executemany(
        "INSERT OR IGNORE INTO target_people (nconst) VALUES (?)",
        [(n,) for n in sorted_nconsts],
    )

    sql_people_meta = f"""
    SELECT
        {people_select_clause(alias='p', prof_alias='pp')}
    FROM people p
    JOIN target_people tp ON tp.nconst = p.nconst
    LEFT JOIN people_professions pp ON pp.nconst = p.nconst
    GROUP BY p.nconst
    """
    cur.execute(sql_people_meta)

    fetched = {}
    for r in cur:
        row_dict = map_person_row(r)
        fetched[row_dict["nconst"]] = row_dict
    return fetched


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_hybrid_cache(cfg: ProjectConfig):
    db_path = Path(cfg.db_path)
    cache_path = Path(cfg.data_dir) / "hybrid_set_cache.pt"

    # Phase 1: Field stats
    mov_ae, people_ae = _init_autoencoders(cfg)

    # Phase 2: Edge bucketing
    logging.info(f"Connecting to {db_path}...")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    movie_data, person_data, movie_head_pops, person_head_pops = (
        _fetch_and_bucket_edges(cur, cfg)
    )

    # Filter to movies that have both cast and director edges
    potential_tconsts = sorted(
        t for t, heads in movie_data.items()
        if heads.get("cast") and heads.get("director")
    )
    logging.info(f"Movie filtering: {len(potential_tconsts)} movies with cast + director.")

    # Phase 3: Global index mappings
    final_tconsts, fetched_rows, tconst_to_idx, sorted_nconsts, nconst_to_idx = (
        _build_global_indices(cur, movie_data, potential_tconsts)
    )
    num_movies = len(final_tconsts)
    num_people = len(sorted_nconsts)

    # Phase 4a: Movie-to-people head vocabularies
    logging.info("Building movie-to-people head vocabularies...")
    head_vocab_sizes, head_mappings, head_local_to_global = (
        _build_head_vocab_mappings(movie_head_pops, nconst_to_idx, num_people)
    )

    # Phase 5a: Stack movie fields and head targets
    stacked_fields, heads_padded, temp_heads_lists = _stack_entity_fields(
        entity_ids=final_tconsts,
        id_to_entity_data=fetched_rows,
        head_data_lookup=lambda t: movie_data[t],
        ae_fields=mov_ae.fields,
        cfg=cfg,
        id_to_global_idx=nconst_to_idx,
        desc="Stacking movies",
    )

    # Phase 6a: Person names
    idx_to_person_name = _fetch_person_names(cur, sorted_nconsts, nconst_to_idx)

    # Restrict person_data to final movies only
    logging.info("Building person-centric movie heads...")
    final_tconsts_set = set(final_tconsts)
    person_data_final = defaultdict(lambda: defaultdict(set))
    person_head_pops_final = defaultdict(set)

    for nconst, head_dict in person_data.items():
        for head, tconsts in head_dict.items():
            for t in tconsts:
                if t in final_tconsts_set:
                    person_data_final[nconst][head].add(t)
                    person_head_pops_final[head].add(t)

    # Phase 6b: Person metadata
    fetched_people_rows = _fetch_person_metadata(cur, sorted_nconsts)

    # Phase 5b: Stack person fields and head targets
    person_stacked_fields, person_heads_padded, _ = _stack_entity_fields(
        entity_ids=sorted_nconsts,
        id_to_entity_data=fetched_people_rows,
        head_data_lookup=lambda n: person_data_final.get(n, {}),
        ae_fields=people_ae.fields,
        cfg=cfg,
        id_to_global_idx=tconst_to_idx,
        desc="Stacking people",
    )

    # Phase 4b: Person-to-movie head vocabularies
    logging.info("Building person-to-movie head vocabularies...")
    person_head_vocab_sizes, person_head_mappings, person_head_local_to_global = (
        _build_head_vocab_mappings(person_head_pops_final, tconst_to_idx, num_movies)
    )

    conn.close()

    # Serialize field configs
    field_configs = {f.name: _field_to_state(f) for f in mov_ae.fields}
    person_field_configs = {f.name: _field_to_state(f) for f in people_ae.fields}

    # Compute average head counts (for logging/diagnostics)
    head_avg_counts = {}
    for head, lists in temp_heads_lists.items():
        total = sum(len(idx_list) for idx_list in lists)
        head_avg_counts[head] = float(total) / max(num_movies, 1)

    payload = {
        "num_movies": num_movies,
        "num_people": num_people,
        "stacked_fields": stacked_fields,
        "heads_padded": heads_padded,
        "head_mappings": head_mappings,
        "head_vocab_sizes": head_vocab_sizes,
        "head_avg_counts": head_avg_counts,
        "idx_to_person_name": idx_to_person_name,
        "field_configs": field_configs,
        "field_names": [f.name for f in mov_ae.fields],
        "head_local_to_global": head_local_to_global,
        "person_stacked_fields": person_stacked_fields,
        "person_heads_padded": person_heads_padded,
        "person_head_mappings": person_head_mappings,
        "person_head_vocab_sizes": person_head_vocab_sizes,
        "person_head_local_to_global": person_head_local_to_global,
        "person_field_configs": person_field_configs,
        "person_field_names": [f.name for f in people_ae.fields],
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
