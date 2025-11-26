# scripts/autoencoder/precompute_joint_cache.py

from __future__ import annotations
import logging
import sqlite3
from pathlib import Path
from typing import List, Dict

import torch
from tqdm import tqdm

from config import ProjectConfig
from .fields import BaseField

logger = logging.getLogger(__name__)


_MOVIE_SQL = """
SELECT primaryTitle,startYear,endYear,runtimeMinutes,
       averageRating,numVotes,
       (SELECT GROUP_CONCAT(genre,',') FROM title_genres WHERE tconst = ?)
FROM titles WHERE tconst = ? LIMIT 1
"""

_PERSON_SQL = """
SELECT primaryName,birthYear,deathYear,
       (SELECT GROUP_CONCAT(profession,',') FROM people_professions WHERE nconst = ?)
FROM people WHERE nconst = ? LIMIT 1
"""


def get_joint_cache_path(cfg: ProjectConfig) -> Path:
    return Path(cfg.data_dir) / "joint_edge_tensors.pt"


def _build_field_storage(num_edges: int, fields: List[BaseField]) -> List[torch.Tensor]:
    tensors: List[torch.Tensor] = []
    for f in fields:
        base = f.get_base_padding_value()
        shape = (num_edges,) + tuple(base.shape)
        tensors.append(torch.empty(shape, dtype=base.dtype))
    return tensors


def _movie_row(cur, cache: Dict[str, Dict], tconst: str):
    if tconst in cache:
        return cache[tconst]
    r = cur.execute(_MOVIE_SQL, (tconst, tconst)).fetchone()
    if r is None:
        row = {
            "tconst": tconst,
            "primaryTitle": None,
            "startYear": None,
            "endYear": None,
            "runtimeMinutes": None,
            "averageRating": None,
            "numVotes": None,
            "genres": [],
        }
    else:
        row = {
            "tconst": tconst,
            "primaryTitle": r[0],
            "startYear": r[1],
            "endYear": r[2],
            "runtimeMinutes": r[3],
            "averageRating": r[4],
            "numVotes": r[5],
            "genres": r[6].split(",") if r[6] else [],
        }
    cache[tconst] = row
    return row


def _person_row(cur, cache: Dict[str, Dict], nconst: str):
    if nconst in cache:
        return cache[nconst]
    r = cur.execute(_PERSON_SQL, (nconst, nconst)).fetchone()
    if r is None:
        row = {
            "primaryName": None,
            "birthYear": None,
            "deathYear": None,
            "professions": None,
        }
    else:
        row = {
            "primaryName": r[0],
            "birthYear": r[1],
            "deathYear": r[2],
            "professions": r[3].split(",") if r[3] else None,
        }
    cache[nconst] = row
    return row


def build_joint_tensor_cache(
    cfg: ProjectConfig,
    db_path: Path,
    movie_fields: List[BaseField],
    person_fields: List[BaseField],
    cache_path: Path | None = None,
) -> Path:
    if cache_path is None:
        cache_path = get_joint_cache_path(cfg)

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("building joint edge tensor cache at %s", cache_path)

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    count_cur = conn.cursor()
    edge_cur = conn.cursor()

    num_edges = count_cur.execute("SELECT COUNT(*) FROM edges;").fetchone()[0]
    if num_edges <= 0:
        conn.close()
        raise RuntimeError("edges table is empty; run scripts/precompute_edges_table.py first")

    logger.info("precomputing tensors for %d edges", num_edges)

    edge_ids = torch.empty(num_edges, dtype=torch.long)
    movie_tensors = _build_field_storage(num_edges, movie_fields)
    person_tensors = _build_field_storage(num_edges, person_fields)

    movie_cur = conn.cursor()
    person_cur = conn.cursor()
    mov_cache: Dict[str, Dict] = {}
    per_cache: Dict[str, Dict] = {}

    edge_cur.execute("SELECT edgeId,tconst,nconst FROM edges ORDER BY edgeId;")

    for idx, (edge_id, tconst, nconst) in enumerate(
        tqdm(edge_cur, total=num_edges, desc="joint edge tensors")
    ):
        edge_ids[idx] = int(edge_id)

        mr = _movie_row(movie_cur, mov_cache, str(tconst))
        pr = _person_row(person_cur, per_cache, str(nconst))

        for j, f in enumerate(movie_fields):
            movie_tensors[j][idx].copy_(f.transform(mr.get(f.name)))

        for j, f in enumerate(person_fields):
            person_tensors[j][idx].copy_(f.transform(pr.get(f.name)))

    conn.close()

    payload = {
        "edge_ids": edge_ids,
        "movie": movie_tensors,
        "person": person_tensors,
        "movie_field_names": [f.name for f in movie_fields],
        "person_field_names": [f.name for f in person_fields],
    }

    logger.info("saving joint edge tensor cache to %s", cache_path)
    torch.save(payload, cache_path)
    logger.info("joint edge tensor cache saved")

    return cache_path


def ensure_joint_tensor_cache(
    cfg: ProjectConfig,
    db_path: Path,
    movie_fields: List[BaseField],
    person_fields: List[BaseField],
) -> Path:
    cache_path = get_joint_cache_path(cfg)
    if cache_path.exists():
        logger.info("joint edge tensor cache found at %s", cache_path)
        return cache_path
    return build_joint_tensor_cache(cfg, db_path, movie_fields, person_fields, cache_path)
