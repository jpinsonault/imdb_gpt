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
    uri = f"file:{db_path}?mode=ro&cache=shared"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)

    cur = conn.cursor()
    pragmas = [
        "PRAGMA query_only = ON;",
        "PRAGMA temp_store = MEMORY;",
        "PRAGMA cache_size = -200000;",
    ]
    for p in pragmas:
        try:
            cur.execute(p)
        except sqlite3.OperationalError:
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
        yield iterable[i: i + batch_size]


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
                movies[tconst]["genres"] = genres_str.split(",") if genres_str else []

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

    # Base
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

    if not nconsts:
        return {}, {}

    X_batch = [torch.stack(xs, dim=0).to(device) for xs in per_field_X]
    Z_batch = people_ae.encoder(X_batch).detach().cpu()

    Zp = {}
    Yp = {}
    for i, nconst in enumerate(nconsts):
        Zp[nconst] = Z_batch[i]
        Yp[nconst] = per_person_targets[i]

    return Zp, Yp


# ---------- canonical spline helpers ----------

def _natural_cubic_spline_m(x: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Compute natural cubic spline second derivatives at knots.

    x: (N,) strictly increasing
    Y: (N,D)
    returns m: (N,D)
    """
    N = x.shape[0]
    D = Y.shape[1]
    if N <= 2:
        return torch.zeros((N, D), dtype=Y.dtype)

    h = x[1:] - x[:-1]  # (N-1,)
    # guard
    h = torch.clamp(h, min=1e-8)

    alpha = torch.zeros((N, D), dtype=Y.dtype)
    for i in range(1, N - 1):
        alpha[i] = (
            3.0 / h[i] * (Y[i + 1] - Y[i])
            - 3.0 / h[i - 1] * (Y[i] - Y[i - 1])
        )

    l = torch.ones(N, dtype=Y.dtype)
    mu = torch.zeros(N, dtype=Y.dtype)
    z = torch.zeros((N, D), dtype=Y.dtype)

    for i in range(1, N - 1):
        l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        l[i] = torch.clamp(l[i], min=1e-8)
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    m = torch.zeros((N, D), dtype=Y.dtype)
    # m[N-1] already zero for natural BC
    for j in range(N - 2, -1, -1):
        m[j] = z[j] - mu[j] * m[j + 1]

    return m


def _eval_spline_on_grid(x: torch.Tensor, Y: torch.Tensor, m: torch.Tensor, t_grid: torch.Tensor) -> torch.Tensor:
    """
    Evaluate natural cubic spline defined by (x, Y, m) at all t in t_grid.

    x: (N,)
    Y: (N,D)
    m: (N,D) second derivatives
    t_grid: (L,)
    returns: (L,D)
    """
    N = x.shape[0]
    D = Y.shape[1]
    L = t_grid.shape[0]

    Z = torch.empty((L, D), dtype=Y.dtype)

    x0 = float(x[0].item())
    xN = float(x[-1].item())

    for li in range(L):
        t = float(t_grid[li].item())

        if t <= x0:
            Z[li] = Y[0]
            continue
        if t >= xN:
            Z[li] = Y[-1]
            continue

        # find segment i with x[i] <= t <= x[i+1]
        # N is tiny (<= 11), so linear scan is fine
        i = 0
        while i < N - 2 and t > float(x[i + 1].item()):
            i += 1

        xi = float(x[i].item())
        xip1 = float(x[i + 1].item())
        h = max(xip1 - xi, 1e-8)

        A = (xip1 - t) / h
        B = (t - xi) / h

        Z[li] = (
            A * Y[i]
            + B * Y[i + 1]
            + ((A ** 3 - A) * m[i] + (B ** 3 - B) * m[i + 1]) * (h ** 2) / 6.0
        )

    return Z


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

    For each title we produce:
      - Mx, My
      - Zt (movie latent)
      - Z_lat_tgts: (L,D) anchors (title + selected people + padded)
      - Z_spline:  (L,D) canonical natural cubic spline along t_grid
      - Yp_tgts:   list[ (L,...) ] people targets per field
      - t_grid:    (L,)
      - time_mask: (L,) 1 for positions with real anchors (title/people)
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

        # collect anchor latents & per-person targets
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

        # valid anchors: title + actual people (capped by num_people)
        people_k = len(steps) - 1  # may be 0
        valid_len = min(people_k + 1, L)

        t_grid = torch.linspace(0.0, 1.0, steps=L)
        time_mask = torch.zeros(L, dtype=torch.float32)
        time_mask[:valid_len] = 1.0

        # build Yp_tgts with dummy slot 0 + people + pad
        Yp_tgts = []
        for fi, f in enumerate(people_fields):
            dummy0 = f.get_base_padding_value()
            seq = [dummy0] + per_field_seq[fi]
            if len(seq) < L:
                seq += [dummy0] * (L - len(seq))
            else:
                seq = seq[:L]
            Yp_tgts.append(torch.stack(seq, dim=0))

        # pad or truncate steps to length L for Z_lat_tgts
        if len(steps) < L:
            last = steps[-1]
            steps = steps + [last] * (L - len(steps))
        else:
            steps = steps[:L]

        Z_lat_tgts = torch.stack(steps, dim=0)

        # ----- canonical spline over the *valid* anchors -----
        # If we have at least 2 anchors, build a natural cubic spline
        # that passes exactly through those latents, then evaluate on t_grid.
        if valid_len >= 2:
            knots_t = t_grid[:valid_len]
            knots_z = Z_lat_tgts[:valid_len]  # (valid_len, D)
            m = _natural_cubic_spline_m(knots_t, knots_z)
            Z_spline = _eval_spline_on_grid(knots_t, knots_z, m, t_grid)
        else:
            # Degenerate: no people anchors → constant path at title latent
            Z_spline = Z_lat_tgts.clone()

        samples.append(
            {
                "Mx": Mx_i,
                "My": My_i,
                "Zt": Zt.clone(),
                "Z_lat_tgts": Z_lat_tgts.clone(),
                "Z_spline": Z_spline.clone(),
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
            people_rows = {
                n: row for n, row in people_rows.items() if n in unique_people
            }
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
