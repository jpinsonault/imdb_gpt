from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import sqlite3
from tqdm import tqdm
import torch
import logging

from scripts.autoencoder.row_ae.imdb import TitlesAutoencoder, PeopleAutoencoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

MOVIE_SQL_BULK = """
SELECT
    t.tconst,
    t.primaryTitle,
    t.startYear,
    t.endYear,
    t.runtimeMinutes,
    t.averageRating,
    t.numVotes,
    (SELECT GROUP_CONCAT(genre, ',') FROM title_genres WHERE tconst = t.tconst)
FROM titles t
WHERE t.tconst IN ({ph})
"""

PEOPLE_SQL_BULK = """
SELECT
    p.nconst,
    p.primaryName,
    p.birthYear,
    p.deathYear,
    (SELECT GROUP_CONCAT(profession, ',') FROM people_professions WHERE nconst = p.nconst)
FROM people p
WHERE p.nconst IN ({ph})
"""

def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")
    conn.execute("PRAGMA mmap_size=268435456;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _write_ids_txt(path: Path, ids: List[str]):
    with open(path, "w", encoding="utf-8") as f:
        for s in ids:
            f.write(f"{s}\n")

def _bulk_fetch(conn: sqlite3.Connection, sql_tmpl: str, ids: List[str], chunk: int) -> Dict[str, tuple]:
    out: Dict[str, tuple] = {}
    cur = conn.cursor()
    for i in tqdm(range(0, len(ids), chunk), desc="bulk fetch", unit="chunk", dynamic_ncols=True):
        batch = ids[i : i + chunk]
        ph = ",".join(["?"] * len(batch))
        sql = sql_tmpl.format(ph=ph)
        cur.execute(sql, batch)
        for r in cur.fetchall():
            out[r[0]] = r
    return out

def _movie_row_from_tuple(r) -> Dict:
    return {
        "tconst": r[0],
        "primaryTitle": r[1],
        "startYear": r[2],
        "endYear": r[3],
        "runtimeMinutes": r[4],
        "averageRating": r[5],
        "numVotes": r[6],
        "genres": r[7].split(",") if r[7] else [],
    }

def _person_row_from_tuple(r) -> Dict:
    return {
        "primaryName": r[1],
        "birthYear": r[2],
        "deathYear": r[3],
        "professions": r[4].split(",") if r[4] else None,
    }

def build_memstore(db_path: str, out_dir: str, chunk: int = 10000):
    out_root = Path(out_dir)
    movies_dir = out_root / "movies"
    people_dir = out_root / "people"
    edges_dir = out_root / "edges"
    _ensure_dir(movies_dir)
    _ensure_dir(people_dir)
    _ensure_dir(edges_dir)

    logging.info("initializing AEs and stats")
    mov_ae = TitlesAutoencoder({"db_path": db_path, "model_dir": ".", "latent_dim": 1, "movie_limit": 1})
    per_ae = PeopleAutoencoder({"db_path": db_path, "model_dir": ".", "latent_dim": 1})
    mov_ae.accumulate_stats()
    mov_ae.finalize_stats()
    per_ae.accumulate_stats()
    per_ae.finalize_stats()

    with _connect(db_path) as conn:
        cur = conn.cursor()

        logging.info("loading distinct ids from edges")
        cur.execute("SELECT DISTINCT tconst FROM edges;")
        movie_ids = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT DISTINCT nconst FROM edges;")
        person_ids = [r[0] for r in cur.fetchall()]
        logging.info(f"counts: movies={len(movie_ids):,} people={len(person_ids):,}")

        movie_idx = {t: i for i, t in enumerate(movie_ids)}
        person_idx = {n: i for i, n in enumerate(person_ids)}

        _write_ids_txt(movies_dir / "ids.txt", movie_ids)
        _write_ids_txt(people_dir / "ids.txt", person_ids)

        movie_field_specs = []
        for f in mov_ae.fields:
            sample = f.transform("").cpu().numpy() if hasattr(f, "tokenizer") else f.transform(0).cpu().numpy()
            movie_field_specs.append({
                "name": f.name,
                "shape": list(sample.shape),
                "dtype": "int32" if sample.dtype.kind in ("i", "u") else "float32",
                "path": f"{f.name}.mm",
            })
        person_field_specs = []
        for f in per_ae.fields:
            sample = f.transform("").cpu().numpy() if hasattr(f, "tokenizer") else f.transform(0).cpu().numpy()
            person_field_specs.append({
                "name": f.name,
                "shape": list(sample.shape),
                "dtype": "int32" if sample.dtype.kind in ("i", "u") else "float32",
                "path": f"{f.name}.mm",
            })

        logging.info("creating memmap files for movie fields")
        for spec in tqdm(movie_field_specs, desc="alloc movies", unit="field", dynamic_ncols=True):
            shape = (len(movie_ids),) + tuple(spec["shape"])
            np.memmap(movies_dir / spec["path"], dtype=spec["dtype"], mode="w+", shape=shape)[:] = 0

        logging.info("creating memmap files for person fields")
        for spec in tqdm(person_field_specs, desc="alloc people", unit="field", dynamic_ncols=True):
            shape = (len(person_ids),) + tuple(spec["shape"])
            np.memmap(people_dir / spec["path"], dtype=spec["dtype"], mode="w+", shape=shape)[:] = 0

        logging.info("bulk fetching rows into RAM caches")
        m_rows = _bulk_fetch(conn, MOVIE_SQL_BULK, movie_ids, chunk)
        p_rows = _bulk_fetch(conn, PEOPLE_SQL_BULK, person_ids, chunk)

        movie_mm = {
            spec["name"]: np.memmap(
                movies_dir / spec["path"],
                dtype=spec["dtype"],
                mode="r+",
                shape=(len(movie_ids),) + tuple(spec["shape"]),
            ) for spec in movie_field_specs
        }
        person_mm = {
            spec["name"]: np.memmap(
                people_dir / spec["path"],
                dtype=spec["dtype"],
                mode="r+",
                shape=(len(person_ids),) + tuple(spec["shape"]),
            ) for spec in person_field_specs
        }

        logging.info("encoding movie fields → memmaps")
        for tconst in tqdm(movie_ids, desc="movies", unit="row", dynamic_ncols=True):
            r = m_rows.get(tconst)
            if r is None:
                continue
            row = _movie_row_from_tuple(r)
            xs = [f.transform(row.get(f.name)) for f in mov_ae.fields]
            xs_np = [x.cpu().numpy() for x in xs]
            i = movie_idx[tconst]
            for f, x in zip(mov_ae.fields, xs_np):
                name = f.name
                if movie_mm[name].dtype == np.int32:
                    movie_mm[name][i] = x.astype(np.int32, copy=False)
                else:
                    movie_mm[name][i] = x.astype(np.float32, copy=False)

        logging.info("encoding person fields → memmaps")
        for nconst in tqdm(person_ids, desc="people", unit="row", dynamic_ncols=True):
            r = p_rows.get(nconst)
            if r is None:
                continue
            row = _person_row_from_tuple(r)
            xs = [f.transform(row.get(f.name)) for f in per_ae.fields]
            xs_np = [x.cpu().numpy() for x in xs]
            i = person_idx[nconst]
            for f, x in zip(per_ae.fields, xs_np):
                name = f.name
                if person_mm[name].dtype == np.int32:
                    person_mm[name][i] = x.astype(np.int32, copy=False)
                else:
                    person_mm[name][i] = x.astype(np.float32, copy=False)

        logging.info("writing edge index arrays")
        cur.execute("SELECT tconst,nconst FROM edges ORDER BY edgeId;")
        pairs = cur.fetchall()
        E = len(pairs)
        movie_idx_mm = np.memmap(edges_dir / "movie_idx.int32.mm", dtype=np.int32, mode="w+", shape=(E,))
        person_idx_mm = np.memmap(edges_dir / "person_idx.int32.mm", dtype=np.int32, mode="w+", shape=(E,))
        for k, (t, n) in enumerate(tqdm(pairs, desc="edges", unit="edge", dynamic_ncols=True)):
            movie_idx_mm[k] = movie_idx.get(t, -1)
            person_idx_mm[k] = person_idx.get(n, -1)

    manifest = {
        "version": 1,
        "counts": {"movies": len(movie_ids), "people": len(person_ids), "edges": E},
        "movies": {"dir": "movies", "ids": "movies/ids.txt", "fields": movie_field_specs},
        "people": {"dir": "people", "ids": "people/ids.txt", "fields": person_field_specs},
        "edges": {"dir": "edges", "movie_idx": "edges/movie_idx.int32.mm", "person_idx": "edges/person_idx.int32.mm"},
    }
    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logging.info(f"memstore built at {out_root}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--chunk", type=int, default=10000)
    args = p.parse_args()
    build_memstore(
        args.db,
        args.out,
        chunk=args.chunk,
    )

if __name__ == "__main__":
    main()
