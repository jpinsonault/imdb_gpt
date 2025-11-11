# scripts/precompute_path_deltas.py

from __future__ import annotations

import argparse
import logging
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm

from config import project_config, ProjectConfig, ensure_dirs
from scripts.autoencoder.ae_loader import _load_frozen_autoencoders
from scripts.sql_filters import movie_where_clause

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("path_deltas_fast")


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
        (name,),
    )
    return cur.fetchone() is not None


def _collect_movie_people_map(
    conn: sqlite3.Connection,
    cfg: ProjectConfig,
    max_people: int,
    movie_limit: int | None,
) -> Tuple[List[str], Dict[str, List[str]]]:
    limit = movie_limit or cfg.path_siren_movie_limit or cfg.movie_limit or None
    principals = cfg.principals_table

    use_edges = _table_exists(conn, "edges")

    movie_to_people: Dict[str, List[str]] = defaultdict(list)

    if use_edges:
        sql = f"""
        SELECT pr.tconst,
               pr.nconst,
               pr.ordering
        FROM edges e
        JOIN {principals} pr
          ON pr.tconst = e.tconst AND pr.nconst = e.nconst
        JOIN people p
          ON p.nconst = pr.nconst
        WHERE p.birthYear IS NOT NULL
        ORDER BY pr.tconst, pr.ordering
        """
        log.info("[path-deltas] using edges + principals for movie->people map")
    else:
        sql = f"""
        SELECT pr.tconst,
               pr.nconst,
               pr.ordering
        FROM {principals} pr
        JOIN titles t ON t.tconst = pr.tconst
        JOIN title_genres g ON g.tconst = t.tconst
        JOIN people p ON p.nconst = pr.nconst
        WHERE {movie_where_clause()}
          AND p.birthYear IS NOT NULL
        GROUP BY pr.tconst, pr.nconst
        ORDER BY pr.tconst, MIN(pr.ordering)
        """
        log.info("[path-deltas] using principals+filters for movie->people map")

    cur = conn.cursor()
    cur.execute(sql)

    last_t = None
    count_movies = 0

    for tconst, nconst, ordering in cur:
        if limit is not None and count_movies >= limit and tconst != last_t:
            break

        if last_t is None or tconst != last_t:
            last_t = tconst
            count_movies += 1

        lst = movie_to_people[tconst]
        if len(lst) < max_people:
            lst.append(nconst)

    cur.close()

    movies = sorted(movie_to_people.keys())
    if limit is not None:
        movies = movies[: int(limit)]

    log.info(
        "[path-deltas] collected %d movies with up to %d people each",
        len(movies),
        max_people,
    )

    return movies, movie_to_people


def _chunked(seq, size: int):
    n = len(seq)
    if n == 0:
        return
    step = max(1, int(size))
    for i in range(0, n, step):
        yield seq[i : i + step]


def _encode_movies_batched(
    conn: sqlite3.Connection,
    mov_ae,
    movie_ids: List[str],
    batch_rows: int,
) -> Dict[str, Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]]:
    device = next(mov_ae.encoder.parameters()).device
    fields = mov_ae.fields

    result: Dict[str, Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]] = {}

    for chunk in tqdm(
        list(_chunked(movie_ids, batch_rows)),
        desc="[path-deltas] encode movies",
        unit="batch",
    ):
        if not chunk:
            continue

        placeholders = ",".join(["?"] * len(chunk))
        sql = f"""
        SELECT t.tconst,
               t.primaryTitle,
               t.startYear,
               t.endYear,
               t.runtimeMinutes,
               t.averageRating,
               t.numVotes,
               (SELECT GROUP_CONCAT(genre,',') FROM title_genres WHERE tconst = t.tconst)
        FROM titles t
        WHERE t.tconst IN ({placeholders})
        """
        rows = conn.execute(sql, chunk).fetchall()
        row_map: Dict[str, Dict[str, Any]] = {}

        for (
            tconst,
            primaryTitle,
            startYear,
            endYear,
            runtimeMinutes,
            averageRating,
            numVotes,
            genres,
        ) in rows:
            row_map[tconst] = {
                "tconst": tconst,
                "primaryTitle": primaryTitle,
                "startYear": startYear,
                "endYear": endYear,
                "runtimeMinutes": runtimeMinutes,
                "averageRating": averageRating,
                "numVotes": numVotes,
                "genres": genres.split(",") if genres else [],
            }

        batch_ids: List[str] = []
        per_field_xs: List[List[torch.Tensor]] = [[] for _ in fields]
        per_field_ys: List[List[torch.Tensor]] = [[] for _ in fields]

        for tconst in chunk:
            row = row_map.get(tconst)
            if row is None:
                continue
            batch_ids.append(tconst)
            for i, f in enumerate(fields):
                x = f.transform(row.get(f.name))
                y = f.transform_target(row.get(f.name))
                per_field_xs[i].append(x)
                per_field_ys[i].append(y)

        if not batch_ids:
            continue

        X_batch = [
            torch.stack(col, dim=0).to(device)
            for col in per_field_xs
        ]

        with torch.no_grad():
            z_batch = mov_ae.encoder(X_batch).cpu()

        for bi, tconst in enumerate(batch_ids):
            Mx = [x[bi].cpu() for x in per_field_xs]
            My = [y[bi].cpu() for y in per_field_ys]
            z = z_batch[bi].cpu()
            result[tconst] = (Mx, My, z)

    log.info("[path-deltas] encoded %d movies", len(result))
    return result


def _encode_people_batched(
    conn: sqlite3.Connection,
    per_ae,
    needed_people: List[str],
    batch_rows: int,
) -> Dict[str, Tuple[List[torch.Tensor], torch.Tensor]]:
    device = next(per_ae.encoder.parameters()).device
    fields = per_ae.fields

    result: Dict[str, Tuple[List[torch.Tensor], torch.Tensor]] = {}

    for chunk in tqdm(
        list(_chunked(needed_people, batch_rows)),
        desc="[path-deltas] encode people",
        unit="batch",
    ):
        if not chunk:
            continue

        placeholders = ",".join(["?"] * len(chunk))
        sql = f"""
        SELECT p.nconst,
               p.primaryName,
               p.birthYear,
               p.deathYear,
               (SELECT GROUP_CONCAT(profession,',')
                FROM people_professions
                WHERE nconst = p.nconst)
        FROM people p
        WHERE p.nconst IN ({placeholders})
        """
        rows = conn.execute(sql, chunk).fetchall()
        row_map: Dict[str, Dict[str, Any]] = {}

        for nconst, primaryName, birthYear, deathYear, profs in rows:
            row_map[nconst] = {
                "primaryName": primaryName,
                "birthYear": birthYear,
                "deathYear": deathYear,
                "professions": profs.split(",") if profs else None,
            }

        batch_ids: List[str] = []
        per_field_ys: List[List[torch.Tensor]] = [[] for _ in fields]

        for nconst in chunk:
            row = row_map.get(nconst)
            if row is None:
                continue
            batch_ids.append(nconst)
            for i, f in enumerate(fields):
                y = f.transform_target(row.get(f.name))
                per_field_ys[i].append(y)

        if not batch_ids:
            continue

        X_batch = [
            torch.stack(col, dim=0).to(device)
            for col in per_field_ys
        ]

        with torch.no_grad():
            z_batch = per_ae.encoder(X_batch).cpu()

        for bi, nconst in enumerate(batch_ids):
            Ys = [y[bi].cpu() for y in per_field_ys]
            z = z_batch[bi].cpu()
            result[nconst] = (Ys, z)

    log.info("[path-deltas] encoded %d people", len(result))
    return result


def _build_samples(
    cfg: ProjectConfig,
    movie_ids: List[str],
    movie_to_people: Dict[str, List[str]],
    mov_data: Dict[str, Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]],
    per_data: Dict[str, Tuple[List[torch.Tensor], torch.Tensor]],
    latent_dim: int,
    people_per_movie: int,
) -> List[Dict[str, Any]]:
    L = int(people_per_movie)
    samples: List[Dict[str, Any]] = []

    for tconst in tqdm(movie_ids, desc="[path-deltas] build samples", unit="movie"):
        if tconst not in mov_data:
            continue

        Mx, My, z_movie = mov_data[tconst]
        people_ids = movie_to_people.get(tconst, [])
        if not isinstance(people_ids, list):
            continue

        z_people_list: List[torch.Tensor] = []
        per_field_targets: List[List[torch.Tensor]] = [
            [] for _ in range(len(next(iter(per_data.values()))[0]))  # type: ignore[index]
        ]

        for nconst in people_ids:
            pd = per_data.get(nconst)
            if pd is None:
                continue
            Ys, z_p = pd
            z_people_list.append(z_p)
            for fi, y in enumerate(Ys):
                per_field_targets[fi].append(y)

        k = min(len(z_people_list), L)
        time_mask = torch.zeros(L, dtype=torch.float32)
        if k > 0:
            time_mask[:k] = 1.0

        if L > 0:
            t_grid = torch.linspace(0.0, 1.0, steps=L)
        else:
            t_grid = torch.zeros(0)

        if L > 0:
            if k > 0:
                seq = z_people_list[:k]
                last = seq[-1]
                while len(seq) < L:
                    seq.append(last)
                Z_lat_tgts = torch.stack(seq, dim=0)
            else:
                Z_lat_tgts = z_movie.unsqueeze(0).expand(L, latent_dim).clone()
        else:
            Z_lat_tgts = torch.zeros(0, latent_dim)

        Yp_tgts: List[torch.Tensor] = []
        for fi in range(len(per_field_targets)):
            base = None
            if per_data:
                any_key = next(iter(per_data.keys()))
                base = per_data[any_key][0][fi].new_tensor(per_data[any_key][0][fi])  # shape prototype
            # safer: pull from first available f.get_base_padding_value at runtime via mov/per_ae if needed.
            seq_vals = per_field_targets[fi][:k]
            if L > 0:
                if len(seq_vals) < L:
                    if base is None:
                        base = seq_vals[0]
                    pad_val = base
                    seq_vals = seq_vals + [pad_val for _ in range(L - len(seq_vals))]
                Yp_tgts.append(torch.stack(seq_vals, dim=0))
            else:
                if base is None:
                    base = torch.tensor(0.0)
                Yp_tgts.append(torch.zeros(0, *base.shape, dtype=base.dtype))

        delta_tgts = torch.zeros(L, latent_dim, dtype=torch.float32)
        if k > 0:
            z_valid = z_people_list[:k]
            delta_tgts[0] = z_valid[0] - z_movie
            for i in range(1, k):
                delta_tgts[i] = z_valid[i] - z_valid[i - 1]

        sample = {
            "tconst": tconst,
            "Mx": [x.cpu() for x in Mx],
            "My": [y.cpu() for y in My],
            "movie_latent": z_movie.cpu(),
            "Z_lat_tgts": Z_lat_tgts.cpu(),
            "Yp_tgts": [y.cpu() for y in Yp_tgts],
            "delta_tgts": delta_tgts.cpu(),
            "time_mask": time_mask.cpu(),
            "t_grid": t_grid.cpu(),
            "num_people": int(k),
        }
        samples.append(sample)

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Precompute fast Path-Siren delta cache (batched encodings)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--people",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--movie-limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=2048,
        help="Rows per encoder batch for movies/people.",
    )
    args = parser.parse_args()

    cfg = project_config
    ensure_dirs(cfg)

    db_path = Path(cfg.db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found at {db_path}")

    out_path = (
        Path(args.output)
        if args.output is not None
        else Path(cfg.data_dir) / "path_siren_deltas.pt"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    people_per_movie = int(args.people or cfg.path_siren_people_count)
    movie_limit = int(args.movie_limit) if args.movie_limit is not None else None
    batch_rows = max(1, int(args.batch_rows))

    log.info("[path-deltas] db=%s", db_path)
    log.info("[path-deltas] output=%s", out_path)
    log.info("[path-deltas] people_per_movie=%d", people_per_movie)
    if movie_limit is not None:
        log.info("[path-deltas] movie_limit=%d", movie_limit)

    mov_ae, per_ae = _load_frozen_autoencoders(cfg)
    mov_ae.encoder.eval()
    per_ae.encoder.eval()
    mov_ae.decoder.eval()
    per_ae.decoder.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mov_ae.encoder.to(device)
    per_ae.encoder.to(device)

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    conn.execute("PRAGMA cache_size = -200000;")
    conn.execute("PRAGMA mmap_size = 268435456;")
    conn.execute("PRAGMA busy_timeout = 5000;")

    movies, movie_to_people = _collect_movie_people_map(
        conn=conn,
        cfg=cfg,
        max_people=people_per_movie,
        movie_limit=movie_limit,
    )
    if not movies:
        raise RuntimeError("[path-deltas] no movies found for cache")

    needed_people: List[str] = []
    seen = set()
    for t in movies:
        for n in movie_to_people.get(t, []):
            if n not in seen:
                seen.add(n)
                needed_people.append(n)

    log.info(
        "[path-deltas] unique people to encode: %d",
        len(needed_people),
    )

    mov_data = _encode_movies_batched(
        conn=conn,
        mov_ae=mov_ae,
        movie_ids=movies,
        batch_rows=batch_rows,
    )

    per_data = _encode_people_batched(
        conn=conn,
        per_ae=per_ae,
        needed_people=needed_people,
        batch_rows=batch_rows,
    )

    if not mov_data:
        raise RuntimeError("[path-deltas] no movie encodings produced")
    if not per_data:
        log.warning("[path-deltas] no person encodings produced; paths will be empty")

    latent_dim = int(getattr(mov_ae, "latent_dim", cfg.latent_dim))

    samples = _build_samples(
        cfg=cfg,
        movie_ids=movies,
        movie_to_people=movie_to_people,
        mov_data=mov_data,
        per_data=per_data,
        latent_dim=latent_dim,
        people_per_movie=people_per_movie,
    )

    conn.close()

    if not samples:
        raise RuntimeError("[path-deltas] no samples generated; nothing to save")

    obj = {
        "version": 2,
        "latent_dim": latent_dim,
        "people_per_movie": int(people_per_movie),
        "samples": samples,
    }

    torch.save(obj, out_path)
    log.info("[path-deltas] saved %d samples to %s", len(samples), out_path)


if __name__ == "__main__":
    main()
