# scripts/joint_edge_sampler.py
import sqlite3, random, tensorflow as tf
from typing import Iterator, Tuple


def make_edge_sampler(
        db_path: str, # Changed type hint to str for consistency with usage
        movie_ae, # Consider adding type hints: TitlesAutoencoder
        person_ae, # Consider adding type hints: PeopleAutoencoder
        batch_size: int, # This parameter is no longer strictly needed here for yielding single items but can be kept for other logic if any
) -> Iterator[
        Tuple[
            Tuple[tf.Tensor, ...],  # movie inputs
            Tuple[tf.Tensor, ...],  # person inputs
        ]
]:
    """
    Mini‑batch generator that reads pre‑filtered (tconst, nconst) pairs
    from the `edges` table created by precompute_edges_table.py.

    Each `yield` returns two tuples of tensors for a *single* movie-person pair.
    The tf.data.Dataset.batch() method will handle the batching.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cur  = conn.cursor()

    cur.execute("SELECT tconst, nconst FROM edges;")
    all_edges = cur.fetchall()
    random.shuffle(all_edges)

    i = 0
    while True:
        if i == len(all_edges):
            random.shuffle(all_edges)
            i = 0
        tconst, nconst = all_edges[i]
        i += 1

        try: # Add try-except for robustness during data fetching/transformation
            m_row = movie_ae.row_by_tconst(tconst)
            p_row = person_ae.row_by_nconst(nconst)

            movie_transformed = tuple(f.transform(m_row.get(f.name)) for f in movie_ae.fields)
            person_transformed = tuple(f.transform(p_row.get(f.name)) for f in person_ae.fields)
            
            yield movie_transformed, person_transformed
        except KeyError as e:
            # This can happen if a tconst or nconst from the edges table is not found by row_by_tconst/nconst
            print(f"Warning: Skipping edge ({tconst}, {nconst}) due to KeyError: {e}") # Or use logging
            continue
        except Exception as e:
            print(f"Warning: Skipping edge ({tconst}, {nconst}) due to an unexpected error: {e}") # Or use logging
            continue