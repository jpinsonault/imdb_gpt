# scripts/precompute_path_siren_cache.py

import sqlite3
from pathlib import Path
from collections import defaultdict

import torch
from tqdm.auto import tqdm

from config import project_config, ensure_dirs
from scripts.autoencoder.ae_loader import _load_frozen_autoencoders


# Tune this based on RAM / GPU
CHUNK_SIZE = 50_000


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    """
    Open SQLite in read-only mode, using pragmas that are safe under concurrency.
    No write-mode PRAGMAs here to avoid 'database is locked' issues.
    """
    # URI read-only; cache=shared lets multiple readers coexist.
    uri = f"file:{db_path}?mode=ro&cache=shared"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)

    cur = conn.cursor()
    pragmas = [
        "PRAGMA query_only = ON;",     # enforce read-only at connection level
        "PRAGMA temp_store = MEMORY;", # keep temp tables in RAM
        "PRAGMA cache_size = -200000;" # ~200k pages in memory (negative = KB)
    ]
    for p in pragmas:
        try:
            cur.execute(p)
        except sqlite3.OperationalError:
            # Some pragmas may be unavailable under certain builds / modes; ignore.
            pass
    cur.close()
    return conn


def _get_tconsts(cur, cfg):
    seed = int(cfg.path_siren_seed)
    movie_limit = getattr(cfg, "path_siren_movie_limit", None)
    movie_limit = None if movie_limit in (None, 0) else int(movie_limit)

    if movie_limit is not None:
        rows = cur.execute(
            """
            SELECT DISTINCT tconst, (movie_hash + ?) AS k
            FROM edges
            ORDER BY k
            LIMIT ?
            """,
            (seed, movie_limit),
        ).fetchall()
        tconsts = [r[0] for r in rows]
    else:
        rows = cur.execute("SELECT DISTINCT tconst FROM edges").fetchall()
        tconsts = [r[0] for r in rows]

    return tconsts


def _batched(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def _load_movies(cur, tconsts):
    """
    Bulk load movie base fields + genres.
    Returns dict[tconst] -> row dict.
    """
    movies = {}

    # Base data
    for batch in _batched(tconsts, 30_000):
        q_marks = ",".join("?" for _ in batch)
        rows = cur.execute(
            f"""
            SELECT tconst,
                   primaryTitle,
                   startYear,
                   endYear,
                   runtimeMinutes,
                   averageRating,
                   numVotes
            FROM titles
            WHERE tconst IN ({q_marks})
            """,
            batch,
        ).fetchall()
        for r in rows:
            movies[r[0]] = {
                "tconst": r[0],
                "primaryTitle": r[1],
                "startYear": r[2],
                "endYear": r[3],
                "runtimeMinutes": r[4],
                "averageRating": r[5],
                "numVotes": r[6],
                "genres": [],
            }

    # Genres
    for batch in _batched(tconsts, 30_000):
        q_marks = ",".join("?" for _ in batch)
        rows = cur.execute(
            f"""
            SELECT tconst, GROUP_CONCAT(genre, ',')
            FROM title_genres
            WHERE tconst IN ({q_marks})
            GROUP BY tconst
            """,
            batch,
        ).fetchall()
        for tconst, genres_str in rows:
            if tconst in movies:
                movies[tconst]["genres"] = (
                    genres_str.split(",") if genres_str else []
                )

    return movies


def _load_title_people(cur, tconsts, principals_table: str, k: int):
    """
    Bulk load ordered people per title (up to k each).
    Returns:
        title_to_people: dict[tconst] -> list[nconst]
        unique_people: set[nconst]
    """
    title_to_people = defaultdict(list)

    for batch in _batched(tconsts, 30_000):
        q_marks = ",".join("?" for _ in batch)
        rows = cur.execute(
            f"""
            SELECT pr.tconst, pr.nconst, pr.ordering
            FROM {principals_table} pr
            JOIN people p ON p.nconst = pr.nconst
            WHERE pr.tconst IN ({q_marks})
              AND p.birthYear IS NOT NULL
            ORDER BY pr.tconst, pr.ordering
            """,
            batch,
        ).fetchall()

        last_t = None
        count_for_t = 0
        for tconst, nconst, ordering in rows:
            if tconst != last_t:
                last_t = tconst
                count_for_t = 0
            if count_for_t < k:
                title_to_people[tconst].append(nconst)
                count_for_t += 1

    unique_people = {n for lst in title_to_people.values() for n in lst}
    return title_to_people, unique_people


def _load_people(cur, nconsts):
    """
    Bulk load people base data + professions.
    Returns dict[nconst] -> row dict.
    """
    people = {}

    # Base data
    for batch in _batched(list(nconsts), 50_000):
        q_marks = ",".join("?" for _ in batch)
        rows = cur.execute(
            f"""
            SELECT nconst,
                   primaryName,
                   birthYear,
                   deathYear
            FROM people
            WHERE nconst IN ({q_marks})
            """,
            batch,
        ).fetchall()
        for r in rows:
            people[r[0]] = {
                "primaryName": r[1],
                "birthYear": r[2],
                "deathYear": r[3],
                "professions": None,
            }

    # Professions
    for batch in _batched(list(nconsts), 50_000):
        q_marks = ",".join("?" for _ in batch)
        rows = cur.execute(
            f"""
            SELECT nconst, GROUP_CONCAT(profession, ',')
            FROM people_professions
            WHERE nconst IN ({q_marks})
            GROUP BY nconst
            """,
            batch,
        ).fetchall()
        for nconst, prof_str in rows:
            if nconst in people:
                people[nconst]["professions"] = (
                    prof_str.split(",") if prof_str else None
                )

    return people


@torch.no_grad()
def _encode_movies(movie_ae, device, movies, tconsts):
    """
    Returns:
        movie_Mx: list[list[Tensor]] aligned with tconsts
        movie_My: list[list[Tensor]] aligned with tconsts
        Zt_dict: dict[tconst] -> Tensor(latent_dim,)
    """
    fields = movie_ae.fields

    per_field_X = [[] for _ in fields]
    per_field_Y = [[] for _ in fields]

    movie_Mx = []
    movie_My = []

    for tconst in tconsts:
        row = movies.get(tconst)
        if row is None:
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

        mx = []
        my = []
        for fi, f in enumerate(fields):
            x = f.transform(row.get(f.name))
            y = f.transform_target(row.get(f.name))
            mx.append(x)
            my.append(y)
            per_field_X[fi].append(x)
            per_field_Y[fi].append(y)

        movie_Mx.append(mx)
        movie_My.append(my)

    X_batch = [torch.stack(xs, dim=0).to(device) for xs in per_field_X]
    Z_batch = movie_ae.encoder(X_batch).detach().cpu()

    Zt_dict = {tconst: Z_batch[i] for i, tconst in enumerate(tconsts)}

    return movie_Mx, movie_My, Zt_dict


@torch.no_grad()
def _encode_people(people_ae, device, people_rows):
    """
    people_rows: dict[nconst] -> dict fields
    Returns:
        Zp: dict[nconst] -> Tensor(D,)
        Yp: dict[nconst] -> list[Tensor]  (per-field targets)
    """
    fields = people_ae.fields
    nconsts = list(people_rows.keys())

    per_field_X = [[] for _ in fields]
    per_person_targets = []

    for nconst in nconsts:
        row = people_rows[nconst]
        xs = []
        ys = []
        for fi, f in enumerate(fields):
            x = f.transform(row.get(f.name))
            y = f.transform_target(row.get(f.name))
            xs.append(x)
            ys.append(y)
            per_field_X[fi].append(x)
        per_person_targets.append(ys)

    X_batch = [torch.stack(xs, dim=0).to(device) for xs in per_field_X]
    Z_batch = people_ae.encoder(X_batch).detach().cpu()

    Zp = {}
    Yp = {}
    for i, nconst in enumerate(nconsts):
        Zp[nconst] = Z_batch[i]
        Yp[nconst] = per_person_targets[i]

    return Zp, Yp


def _build_samples_for_chunk(
    tconsts,
    num_people,
    movie_Mx,
    movie_My,
    Zt_dict,
    title_to_people,
    Zp,
    Yp,
    people_fields,
):
    """
    Assemble final samples for this chunk using precomputed latents and targets.
    """
    samples = []
    L = num_people + 1

    idx_for_tconst = {t: i for i, t in enumerate(tconsts)}

    for tconst in tconsts:
        i = idx_for_tconst[tconst]

        Mx_i = [x.clone() for x in movie_Mx[i]]
        My_i = [y.clone() for y in movie_My[i]]
        Zt = Zt_dict[tconst]

        people = title_to_people.get(tconst, [])
        steps = [Zt]
        per_field_seq = [[] for _ in people_fields]

        for nconst in people[:num_people]:
            z_p = Zp.get(nconst)
            ys = Yp.get(nconst)
            if z_p is None or ys is None:
                continue
            steps.append(z_p)
            for fi, y in enumerate(ys):
                per_field_seq[fi].append(y)

        people_k = len(steps) - 1
        valid_len = min(people_k + 1, L)

        t_grid = torch.linspace(0.0, 1.0, steps=L)
        time_mask = torch.zeros(L, dtype=torch.float32)
        time_mask[:valid_len] = 1.0

        Yp_tgts = []
        for fi, f in enumerate(people_fields):
            dummy0 = f.get_base_padding_value()
            seq = [dummy0] + per_field_seq[fi]
            if len(seq) < L:
                seq += [dummy0] * (L - len(seq))
            else:
                seq = seq[:L]
            Yp_tgts.append(torch.stack(seq, dim=0))

        if len(steps) < L:
            last = steps[-1]
            steps += [last] * (L - len(steps))
        else:
            steps = steps[:L]
        Z_lat_tgts = torch.stack(steps, dim=0)

        samples.append(
            {
                "Mx": Mx_i,
                "My": My_i,
                "Zt": Zt.clone(),
                "Z_lat_tgts": Z_lat_tgts.clone(),
                "Yp_tgts": [t.clone() for t in Yp_tgts],
                "t_grid": t_grid.clone(),
                "time_mask": time_mask.clone(),
            }
        )

    return samples


@torch.no_grad()
def main():
    cfg = project_config
    ensure_dirs(cfg)

    db_path = Path(cfg.db_path)
    if not db_path.exists():
        raise SystemExit(f"[precompute] DB not found at {db_path}")

    cache_path = Path(cfg.data_dir) / "path_siren_cache.pt"

    print(f"[precompute] loading frozen joint autoencoders…")
    movie_ae, people_ae = _load_frozen_autoencoders(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    movie_ae.encoder.to(device).eval()
    people_ae.encoder.to(device).eval()

    num_people = int(cfg.path_siren_people_count)

    print(f"[precompute] connecting to {db_path} (read-only)")
    conn = _connect_readonly(db_path)
    cur = conn.cursor()

    print("[precompute] scanning tconsts from edges…")
    all_tconsts = _get_tconsts(cur, cfg)
    if not all_tconsts:
        raise SystemExit("[precompute] no titles found from edges")

    print(f"[precompute] titles: {len(all_tconsts):,}")
    print(f"[precompute] people per title: {num_people}")
    print(f"[precompute] chunk size: {CHUNK_SIZE:,}")

    all_samples = []
    chunk_iter = list(_batched(all_tconsts, CHUNK_SIZE))

    pbar = tqdm(chunk_iter, desc="[precompute] chunks", unit="chunk", dynamic_ncols=True)
    for tconst_chunk in pbar:
        movies = _load_movies(cur, tconst_chunk)

        title_to_people, unique_people = _load_title_people(
            cur, tconst_chunk, cfg.principals_table, num_people
        )

        if unique_people:
            people_rows = _load_people(cur, unique_people)
            people_rows = {n: row for n, row in people_rows.items() if n in unique_people}
            Zp, Yp = _encode_people(people_ae, device, people_rows)
        else:
            Zp, Yp = {}, {}

        movie_Mx, movie_My, Zt_dict = _encode_movies(
            movie_ae, device, movies, tconst_chunk
        )

        samples = _build_samples_for_chunk(
            tconst_chunk,
            num_people,
            movie_Mx,
            movie_My,
            Zt_dict,
            title_to_people,
            Zp,
            Yp,
            people_ae.fields,
        )

        all_samples.extend(samples)

    conn.close()

    print(f"[precompute] writing {len(all_samples):,} samples to {cache_path}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "meta": {
                "num_titles": len(all_samples),
                "num_people": num_people,
                "latent_dim": int(getattr(movie_ae, "latent_dim", 0)),
            },
            "samples": all_samples,
        },
        cache_path,
    )

    print("[precompute] done.")


if __name__ == "__main__":
    main()
