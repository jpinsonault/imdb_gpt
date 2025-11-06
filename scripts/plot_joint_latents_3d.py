# scripts/plot_joint_latents_3d.py

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from config import project_config, ensure_dirs
from scripts.autoencoder.ae_loader import _load_frozen_autoencoders
from scripts.sql_filters import (
    movie_from_join,
    movie_where_clause,
    movie_group_by,
    movie_having,
    people_from_join,
    people_where_clause,
    people_group_by,
    people_having,
)


def _pca3(x):
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean(axis=0, keepdims=True)
    x0 = x - mu
    _, _, vt = np.linalg.svd(x0, full_matrices=False)
    w = vt[:3].T
    y = x0 @ w
    return y, w, mu


def _movie_sql_fields():
    return """
        SELECT
            t.tconst,
            t.primaryTitle,
            t.startYear,
            t.endYear,
            t.runtimeMinutes,
            t.averageRating,
            t.numVotes,
            GROUP_CONCAT(g.genre, ',')
    """


def _movie_sql_core():
    return f"""
        {_movie_sql_fields()}
        {movie_from_join()}
        WHERE {movie_where_clause()}
        {movie_group_by()}
        {movie_having()}
    """


def _movie_sql_count():
    return f"""
        SELECT COUNT(*)
        FROM (
            SELECT t.tconst
            {movie_from_join()}
            WHERE {movie_where_clause()}
            {movie_group_by()}
            {movie_having()}
        ) q
    """


def _people_sql_fields():
    return """
        SELECT
            p.nconst,
            p.primaryName,
            p.birthYear,
            p.deathYear,
            GROUP_CONCAT(pp.profession, ',')
    """


def _people_sql_core():
    return f"""
        {_people_sql_fields()}
        {people_from_join()}
        WHERE {people_where_clause()}
        {people_group_by()}
        {people_having()}
    """


def _people_sql_count():
    return f"""
        SELECT COUNT(*)
        FROM (
            SELECT 1
            {people_from_join()}
            WHERE {people_where_clause()}
            {people_group_by()}
            {people_having()}
        ) q
    """


@torch.no_grad()
def _encode_movies_all(ae, db_path: str, batch_size: int, device):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    total = conn.execute(_movie_sql_count()).fetchone()[0]
    if int(total) == 0:
        conn.close()
        return np.zeros((0, ae.latent_dim), dtype=np.float32), [], []
    latents = []
    labels = []
    ids = []
    steps = (total + batch_size - 1) // batch_size
    for i in tqdm(range(steps), total=steps, desc="encode movies", dynamic_ncols=True):
        off = i * batch_size
        cur = conn.execute(_movie_sql_core() + " LIMIT ? OFFSET ?;", (batch_size, off))
        rows = cur.fetchall()
        xs = []
        for f in ae.fields:
            if f.name == "primaryTitle":
                col = [f.transform(r[1]) for r in rows]
            elif f.name == "startYear":
                col = [f.transform(r[2]) for r in rows]
            else:
                col = [f.get_base_padding_value() for _ in rows]
            xs.append(torch.stack(col, dim=0).to(device))
        z = ae.encoder(xs)
        latents.append(z.detach().cpu().numpy())
        for r in rows:
            tconst = str(r[0] or "")
            title = str(r[1] or "")
            year = str(r[2] or "")
            ids.append(tconst)
            labels.append(title if not year else f"{title} ({year})")
    conn.close()
    return np.concatenate(latents, axis=0), labels, ids


@torch.no_grad()
def _encode_people_all(ae, db_path: str, batch_size: int, device):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    total = conn.execute(_people_sql_count()).fetchone()[0]
    if int(total) == 0:
        conn.close()
        return np.zeros((0, ae.latent_dim), dtype=np.float32), [], []
    latents = []
    labels = []
    ids = []
    steps = (total + batch_size - 1) // batch_size
    for i in tqdm(range(steps), total=steps, desc="encode people", dynamic_ncols=True):
        off = i * batch_size
        cur = conn.execute(_people_sql_core() + " LIMIT ? OFFSET ?;", (batch_size, off))
        rows = cur.fetchall()
        xs = []
        for f in ae.fields:
            if f.name == "primaryName":
                col = [f.transform(r[1]) for r in rows]
            elif f.name == "birthYear":
                col = [f.transform(r[2]) for r in rows]
            else:
                col = [f.get_base_padding_value() for _ in rows]
            xs.append(torch.stack(col, dim=0).to(device))
        z = ae.encoder(xs)
        latents.append(z.detach().cpu().numpy())
        for r in rows:
            nconst = str(r[0] or "")
            name = str(r[1] or "")
            year = str(r[2] or "")
            ids.append(nconst)
            labels.append(name if not year else f"{name} ({year})")
    conn.close()
    return np.concatenate(latents, axis=0), labels, ids


def _fit_gmm_or_kmeans(x, n_components: int, seed: int):
    try:
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=n_components, covariance_type="diag", random_state=seed, max_iter=500)
        gmm.fit(x)
        labels = gmm.predict(x)
        return labels
    except Exception:
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_components, n_init=10, random_state=seed, max_iter=500)
            labels = kmeans.fit_predict(x)
            return labels
        except Exception:
            return np.zeros((x.shape[0],), dtype=np.int32)


def _palette_movies(k):
    base = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
        "#3182bd", "#e6550d", "#31a354", "#756bb1", "#636363",
    ]
    if k <= len(base):
        return base[:k]
    rep = []
    for i in range(k):
        rep.append(base[i % len(base)])
    return rep


def _palette_people(k):
    base = [
        "#0d0887", "#6a00a8", "#b12a90", "#e16462", "#fca636",
        "#0a58ca", "#198754", "#dc3545", "#6c757d", "#fd7e14",
        "#20c997", "#6610f2", "#d63384", "#0dcaf0", "#ffc107",
        "#adb5bd", "#845ef7", "#12b886", "#e8590c", "#868e96",
    ]
    if k <= len(base):
        return base[:k]
    rep = []
    for i in range(k):
        rep.append(base[i % len(base)])
    return rep


def _colors_from_labels(labels, palette):
    colors = []
    m = len(palette)
    for j in labels:
        c = palette[int(j) % m]
        colors.append(c)
    return colors


def _load_edges_for_hover(db_path: str, movie_ids, person_ids, max_edges_per_node: int):
    m_index = {tconst: i for i, tconst in enumerate(movie_ids)}
    p_index = {nconst: i for i, nconst in enumerate(person_ids)}
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cur = conn.cursor()
    edges = []
    q = "SELECT tconst, nconst FROM edges"
    count = cur.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    max_show = 2000000
    taken = 0
    per_m = {}
    per_p = {}
    for tconst, nconst in tqdm(cur.execute(q), total=count, unit="edge", dynamic_ncols=True, desc="scan edges"):
        if tconst in m_index and nconst in p_index:
            mi = m_index[tconst]
            pi = p_index[nconst]
            if per_m.get(mi, 0) < max_edges_per_node and per_p.get(pi, 0) < max_edges_per_node:
                edges.append([mi, pi])
                per_m[mi] = per_m.get(mi, 0) + 1
                per_p[pi] = per_p.get(pi, 0) + 1
                taken += 1
                if taken >= max_show:
                    break
    conn.close()
    return edges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", type=int, default=20)
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--azim", type=float, default=35.0)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-edges-per-node", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out-json", type=str, default="data/latents_3d.json")
    args = parser.parse_args()

    ensure_dirs(project_config)
    mov_ae, per_ae = _load_frozen_autoencoders(project_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mov_ae.encoder.to(device).eval()
    per_ae.encoder.to(device).eval()

    batch_size = int(args.batch_size)
    out_json = Path(args.out_json)

    m_lat, m_labels, m_ids = _encode_movies_all(mov_ae, project_config.db_path, batch_size, device)
    p_lat, p_labels, p_ids = _encode_people_all(per_ae, project_config.db_path, batch_size, device)

    if m_lat.size == 0 and p_lat.size == 0:
        print("no latents to export")
        return

    if m_lat.size == 0:
        all_lat = p_lat
    elif p_lat.size == 0:
        all_lat = m_lat
    else:
        all_lat = np.concatenate([m_lat, p_lat], axis=0)

    tqdm.write("projecting to 3D with PCA ...")
    coords3, basis, mean_vec = _pca3(all_lat)
    n_m = m_lat.shape[0]
    coords_m = coords3[:n_m] if n_m > 0 else np.zeros((0, 3), dtype=np.float32)
    coords_p = coords3[n_m:] if coords3.shape[0] > n_m else np.zeros((0, 3), dtype=np.float32)

    m_k = int(args.components)
    p_k = int(args.components)
    m_labels_cl = _fit_gmm_or_kmeans(m_lat if m_lat.size else np.zeros((0, 3)), m_k, args.seed) if m_lat.size else np.zeros((0,), dtype=np.int32)
    p_labels_cl = _fit_gmm_or_kmeans(p_lat if p_lat.size else np.zeros((0, 3)), p_k, args.seed) if p_lat.size else np.zeros((0,), dtype=np.int32)

    colors_m = _colors_from_labels(m_labels_cl, _palette_movies(m_k)) if m_lat.size else []
    colors_p = _colors_from_labels(p_labels_cl, _palette_people(p_k)) if p_lat.size else []

    edges_pairs = _load_edges_for_hover(
        project_config.db_path,
        m_ids,
        p_ids,
        max_edges_per_node=int(args.max_edges_per_node),
    )

    payload = {
        "coords_m": coords_m.tolist(),
        "coords_p": coords_p.tolist(),
        "labels_m": m_labels,
        "labels_p": p_labels,
        "colors_m": colors_m,
        "colors_p": colors_p,
        "edges": edges_pairs,
        "meta": {
            "n_movies": int(coords_m.shape[0]),
            "n_people": int(coords_p.shape[0]),
            "point_size": float(args.point_size),
            "alpha": float(args.alpha),
            "elev": float(args.elev),
            "azim": float(args.azim),
        }
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"), ensure_ascii=False)

    print(f"saved data to {out_json}")


if __name__ == "__main__":
    main()
