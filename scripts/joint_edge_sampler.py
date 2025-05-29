# scripts/joint_edge_sampler.py
import sqlite3, random, tensorflow as tf
from typing import Iterator, Tuple


def make_edge_sampler(
        db_path: str,
        movie_ae,
        person_ae,
        batch_size: int,
) -> Iterator[
        Tuple[
            Tuple[tf.Tensor, ...],  # movie inputs
            Tuple[tf.Tensor, ...],  # person inputs
        ]
]:
    """
    Mini‑batch generator that reads pre‑filtered (tconst, nconst) pairs
    from the `edges` table created by precompute_edges_table.py.

    Each `yield` returns two tuples of tensors with *batch‑axis first*,
    ready for the joint autoencoder.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cur  = conn.cursor()

    # Pull the full list once and shuffle in RAM.
    # (If RAM is a concern, load in chunks; for ≤ 40 M edges this is fine.)
    cur.execute("SELECT tconst, nconst FROM edges;")
    all_edges = cur.fetchall()
    random.shuffle(all_edges)

    i = 0
    while True:
        mb_m, mb_p = [], []
        for _ in range(batch_size):
            if i == len(all_edges):
                random.shuffle(all_edges)
                i = 0
            tconst, nconst = all_edges[i]
            i += 1

            m_row = movie_ae.row_by_tconst(tconst)
            p_row = person_ae.row_by_nconst(nconst)

            mb_m.append(tuple(f.transform(m_row.get(f.name))  for f in movie_ae.fields))
            mb_p.append(tuple(f.transform(p_row.get(f.name)) for f in person_ae.fields))

        # stack along batch axis
        yield (
            tuple(map(tf.stack, zip(*mb_m))),
            tuple(map(tf.stack, zip(*mb_p))),
        )
