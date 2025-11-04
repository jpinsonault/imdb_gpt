# scripts/plot_joint_latents_3d_interactive.py

import sqlite3
from pathlib import Path

import numpy as np
import torch
import plotly.graph_objects as go
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
        return np.zeros((0, ae.latent_dim), dtype=np.float32), []
    latents = []
    labels = []
    steps = (total + batch_size - 1) // batch_size
    for i in tqdm(range(steps), total=steps, desc="encode movies", dynamic_ncols=True):
        off = i * batch_size
        cur = conn.execute(_movie_sql_core() + " LIMIT ? OFFSET ?;", (batch_size, off))
        rows = cur.fetchall()
        xs = []
        cols = []
        for f in ae.fields:
            col = [f.transform(r[1] if f.name == "primaryTitle" else r[2]) for r in rows]
            xs.append(torch.stack(col, dim=0).to(device))
            cols.append(col)
        z = ae.encoder(xs)
        latents.append(z.detach().cpu().numpy())
        for r in rows:
            title = str(r[1] or "")
            year = str(r[2] or "")
            labels.append(title if not year else f"{title} ({year})")
    conn.close()
    return np.concatenate(latents, axis=0), labels


@torch.no_grad()
def _encode_people_all(ae, db_path: str, batch_size: int, device):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    total = conn.execute(_people_sql_count()).fetchone()[0]
    if int(total) == 0:
        conn.close()
        return np.zeros((0, ae.latent_dim), dtype=np.float32), []
    latents = []
    labels = []
    steps = (total + batch_size - 1) // batch_size
    for i in tqdm(range(steps), total=steps, desc="encode people", dynamic_ncols=True):
        off = i * batch_size
        cur = conn.execute(_people_sql_core() + " LIMIT ? OFFSET ?;", (batch_size, off))
        rows = cur.fetchall()
        xs = []
        for f in ae.fields:
            if f.name == "primaryName":
                col = [f.transform(r[0]) for r in rows]
            elif f.name == "birthYear":
                col = [f.transform(r[1]) for r in rows]
            else:
                col = [f.get_base_padding_value() for _ in rows]
            xs.append(torch.stack(col, dim=0).to(device))
        z = ae.encoder(xs)
        latents.append(z.detach().cpu().numpy())
        for r in rows:
            name = str(r[0] or "")
            year = str(r[1] or "")
            labels.append(name if not year else f"{name} ({year})")
    conn.close()
    return np.concatenate(latents, axis=0), labels


def _plot(coords_m, coords_p, labels_m, labels_p, out_html: Path, point_size: float, alpha: float, elev: float, azim: float):
    fig = go.Figure()
    if coords_m.size > 0:
        fig.add_trace(
            go.Scatter3d(
                x=coords_m[:, 0],
                y=coords_m[:, 1],
                z=coords_m[:, 2],
                mode="markers",
                name="movies",
                text=labels_m,
                hoverinfo="text",
                marker=dict(size=point_size, opacity=alpha),
            )
        )
    if coords_p.size > 0:
        fig.add_trace(
            go.Scatter3d(
                x=coords_p[:, 0],
                y=coords_p[:, 1],
                z=coords_p[:, 2],
                mode="markers",
                name="people",
                text=labels_p,
                hoverinfo="text",
                marker=dict(symbol="diamond", size=point_size, opacity=alpha),
            )
        )
    fig.update_layout(
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(itemsizing="constant"),
        title="Joint Latent Space (PCA -> 3D)",
    )
    fig.update_scenes(
        camera_eye=dict(
            x=float(np.cos(np.deg2rad(azim)) * np.cos(np.deg2rad(elev))),
            y=float(np.sin(np.deg2rad(azim)) * np.cos(np.deg2rad(elev))),
            z=float(np.sin(np.deg2rad(elev))),
        )
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn", full_html=True)
    return out_html


def main():
    ensure_dirs(project_config)
    mov_ae, per_ae = _load_frozen_autoencoders(project_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mov_ae.encoder.to(device).eval()
    per_ae.encoder.to(device).eval()

    batch_size = 2048
    out_html = Path("data/latents_3d.html")
    out_npz = Path("data/latents_3d.npz")
    elev = 20.0
    azim = 35.0
    point_size = 3.0
    alpha = 0.7

    m_lat, m_labels = _encode_movies_all(mov_ae, project_config.db_path, batch_size, device)
    p_lat, p_labels = _encode_people_all(per_ae, project_config.db_path, batch_size, device)

    if m_lat.size == 0 and p_lat.size == 0:
        print("no latents to plot")
        return

    if m_lat.size == 0:
        all_lat = p_lat
    elif p_lat.size == 0:
        all_lat = m_lat
    else:
        all_lat = np.concatenate([m_lat, p_lat], axis=0)

    coords3, basis, mean_vec = _pca3(all_lat)
    n_m = m_lat.shape[0]
    coords_m = coords3[:n_m] if n_m > 0 else np.zeros((0, 3), dtype=np.float32)
    coords_p = coords3[n_m:] if coords3.shape[0] > n_m else np.zeros((0, 3), dtype=np.float32)

    saved = _plot(coords_m, coords_p, m_labels, p_labels, out_html, point_size, alpha, elev, azim)
    print(f"saved interactive html to {saved}")

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    kinds_m = np.zeros((coords_m.shape[0],), dtype=np.int32)
    kinds_p = np.ones((coords_p.shape[0],), dtype=np.int32)
    kinds = np.concatenate([kinds_m, kinds_p]) if kinds_m.size or kinds_p.size else np.zeros((0,), dtype=np.int32)
    labels = np.array(m_labels + p_labels, dtype=object)
    np.savez_compressed(out_npz, coords=coords3, kinds=kinds, labels=labels, basis=basis, mean=mean_vec)
    print(f"saved data dump to {out_npz}")


if __name__ == "__main__":
    main()
