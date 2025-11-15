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


def _tsne2(x, seed: int, perplexity: float, n_iter: int, lr: float):
    from sklearn.manifold import TSNE
    import inspect
    x = np.asarray(x, dtype=np.float32)
    if x.shape[0] <= 1:
        return np.zeros((x.shape[0], 2), dtype=np.float32)
    p = max(5.0, min(perplexity, max(5.0, (x.shape[0] - 1) / 3)))

    d = x.shape[1]
    k = 32
    if d > k:
        pbar = tqdm(total=3, desc="reduce to 32D (PCA)", dynamic_ncols=True)
        mu = x.mean(axis=0, keepdims=True).astype(np.float32)
        pbar.update(1)
        x0 = (x - mu).astype(np.float32)
        _, _, vt = np.linalg.svd(x0, full_matrices=False)
        pbar.update(1)
        w = vt[:k].T.astype(np.float32)
        x = x0 @ w
        pbar.update(1)
        pbar.close()

    sig = inspect.signature(TSNE.__init__).parameters
    kwargs = {
        "n_components": 2,
        "perplexity": p,
        "learning_rate": lr,
        "init": "pca",
        "random_state": seed,
        "metric": "euclidean",
        "verbose": 0,
        "method": "barnes_hut",
        "angle": 0.5,
    }
    if "n_iter" in sig:
        kwargs["n_iter"] = int(n_iter)
    if "square_distances" in sig:
        kwargs["square_distances"] = True
    if "n_jobs" in sig:
        kwargs["n_jobs"] = -1

    tsne = TSNE(**kwargs)
    y = tsne.fit_transform(x)
    return y.astype(np.float32)


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


def _encode_movies_all(ae, db_path: str, batch_size: int, device):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    total = conn.execute(_movie_sql_count()).fetchone()[0]
    if int(total) == 0:
        conn.close()
        return np.zeros((0, ae.latent_dim), dtype=np.float32), [], []
    latents = []
    labels = []
    ids = []
    cur = conn.cursor()
    cur.execute(_movie_sql_core() + ";")
    with torch.inference_mode():
        pbar = tqdm(total=total, desc="encode movies", dynamic_ncols=True)
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
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
            pbar.update(len(rows))
        pbar.close()
    conn.close()
    return np.concatenate(latents, axis=0), labels, ids


def _encode_people_all(ae, db_path: str, batch_size: int, device):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    total = conn.execute(_people_sql_count()).fetchone()[0]
    if int(total) == 0:
        conn.close()
        return np.zeros((0, ae.latent_dim), dtype=np.float32), [], []
    latents = []
    labels = []
    ids = []
    cur = conn.cursor()
    cur.execute(_people_sql_core() + ";")
    with torch.inference_mode():
        pbar = tqdm(total=total, desc="encode people", dynamic_ncols=True)
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
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
            pbar.update(len(rows))
        pbar.close()
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


def _load_movie_to_people(db_path: str, movie_ids, person_ids, max_edges_per_movie: int):
    m_index = {tconst: i for i, tconst in enumerate(movie_ids)}
    p_index = {nconst: i for i, nconst in enumerate(person_ids)}
    m2p = [[] for _ in range(len(movie_ids))]
    if not m_index or not p_index:
        return m2p
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cur = conn.cursor()
    count = cur.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    q = "SELECT tconst, nconst FROM edges"
    per_m = {}
    for tconst, nconst in tqdm(cur.execute(q), total=count, unit="edge", dynamic_ncols=True, desc="scan edges"):
        if tconst in m_index and nconst in p_index:
            mi = m_index[tconst]
            if per_m.get(mi, 0) < max_edges_per_movie:
                m2p[mi].append(p_index[nconst])
                per_m[mi] = per_m.get(mi, 0) + 1
    conn.close()
    return m2p


def _build_links3d(coords_m, coords_p, movie_to_people, max_links_total):
    x = []
    y = []
    z = []
    pairs = []
    taken = 0
    for mi, plist in enumerate(movie_to_people):
        if mi >= coords_m.shape[0]:
            continue
        mx, my, mz = coords_m[mi].tolist()
        for pj in plist:
            if pj >= coords_p.shape[0]:
                continue
            px, py, pz = coords_p[pj].tolist()
            x.extend([mx, px, None])
            y.extend([my, py, None])
            z.extend([mz, pz, None])
            pairs.append([mi, pj])
            taken += 1
            if max_links_total and taken >= max_links_total:
                return x, y, z, pairs
    return x, y, z, pairs


def _write_tsne_html(out_html: Path, coords_m_2d, coords_p_2d, labels_m, labels_p, colors_m, colors_p, movie_to_people):
    try:
        import plotly.graph_objects as go
        import plotly.offline as po
    except Exception:
        print("plotly not available; skipping HTML")
        return

    xm = coords_m_2d[:, 0].tolist() if coords_m_2d.size else []
    ym = coords_m_2d[:, 1].tolist() if coords_m_2d.size else []
    xp = coords_p_2d[:, 0].tolist() if coords_p_2d.size else []
    yp = coords_p_2d[:, 1].tolist() if coords_p_2d.size else []

    hover_m = labels_m
    hover_p = labels_p

    traces = []

    if xm:
        traces.append(go.Scattergl(
            x=xm,
            y=ym,
            mode="markers",
            name="Movies",
            text=hover_m,
            hoverinfo="text",
            marker=dict(size=6, opacity=0.9, color=colors_m, symbol="square")
        ))

    if xp:
        traces.append(go.Scattergl(
            x=xp,
            y=yp,
            mode="markers",
            name="People",
            text=hover_p,
            hoverinfo="text",
            marker=dict(size=5, opacity=0.85, color=colors_p, symbol="circle")
        ))

    layout = go.Layout(
        title="t-SNE Movies & People",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=10, r=10, t=40, b=10),
    )

    fig = go.Figure(data=traces, layout=layout)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    po.plot(fig, filename=str(out_html), auto_open=False, include_plotlyjs="cdn")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", type=int, default=20)
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--azim", type=float, default=35.0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-edges-per-node", type=int, default=20)
    parser.add_argument("--max-links-total", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out-json", type=str, default="data/latents_3d.json")
    parser.add_argument("--out-html", type=str, default="data/latents_tsne.html")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-iter", type=int, default=1000)
    parser.add_argument("--tsne-lr", type=float, default=200.0)
    args = parser.parse_args()

    ensure_dirs(project_config)
    mov_ae, per_ae = _load_frozen_autoencoders(project_config)

    dev_str = str(args.device).lower()
    if dev_str == "cpu":
        device = torch.device("cpu")
        if args.cpu_threads and args.cpu_threads > 0:
            torch.set_num_threads(int(args.cpu_threads))
            torch.set_num_interop_threads(max(1, int(args.cpu_threads) // 2))
    elif dev_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    mov_ae.encoder.to(device).eval()
    per_ae.encoder.to(device).eval()

    batch_size = int(args.batch_size)
    out_json = Path(args.out_json)
    out_html = Path(args.out_html)

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

    movie_to_people = _load_movie_to_people(
        project_config.db_path,
        m_ids,
        p_ids,
        max_edges_per_movie=int(args.max_edges_per_node),
    )

    payload = {
        "coords_m": coords_m.tolist(),
        "coords_p": coords_p.tolist(),
        "labels_m": m_labels,
        "labels_p": p_labels,
        "colors_m": colors_m,
        "colors_p": colors_p,
        "movie_to_people": movie_to_people,
        "links_3d": {
            "x": [],
            "y": [],
            "z": [],
            "pairs": []
        },
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

    try:
        tqdm.write("computing t-SNE 2D ...")
        tsne_all = _tsne2(all_lat, args.seed, args.tsne_perplexity, args.tsne_iter, args.tsne_lr)
        tsne_m = tsne_all[:n_m] if n_m > 0 else np.zeros((0, 2), dtype=np.float32)
        tsne_p = tsne_all[n_m:] if tsne_all.shape[0] > n_m else np.zeros((0, 2), dtype=np.float32)
        _write_tsne_html(out_html, tsne_m, tsne_p, m_labels, p_labels, colors_m, colors_p, movie_to_people)
        print(f"saved t-SNE html to {out_html}")
    except Exception as e:
        print(f"tsne/html generation failed: {e}")

    print(f"saved data to {out_json}")

if __name__ == "__main__":
    main()
