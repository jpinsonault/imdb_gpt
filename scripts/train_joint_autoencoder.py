# scripts/train_joint_autoencoder.py
"""
Joint‑embedding trainer that learns **one latent space** shared by
*movies* and the *people who worked on them*.

•  If you pass  --warm   we load   TitlesAutoencoder   and   PeopleAutoencoder
   from <model_dir>/TitlesAutoencoder / PeopleAutoencoder  and keep finetuning
   them jointly.  
•  If you omit  --warm   we build them from scratch and train everything end‑to‑end.

During every step we:
  1)  reconstruct the movie row  (all field decoders)
  2)  reconstruct the person row (all field decoders)
  3)  push the two latent vectors together with an InfoNCE loss
      (movie ↔ matching‑person = positive, everything else in the batch = negative)

The end result is that
    – similar movies are near the people who made them
    – we still have per‑row reconstruction, so you can decode
      either entity type back to its original columns.
"""
from __future__ import annotations
import logging
import argparse, os, random, sqlite3
from pathlib import Path
from typing import Dict, Any, Tuple

import tensorflow as tf
from tensorflow.keras import layers as KL
from tensorflow.keras import ops

from config import project_config
from scripts.autoencoder.imdb_row_autoencoders import TitlesAutoencoder, PeopleAutoencoder
from scripts.autoencoder.training_callbacks import JointReconstructionCallback, TensorBoardPerBatchLoggingCallback
logging.basicConfig(level=logging.INFO)

##########################################################################
# helpers
##########################################################################
def cosine_similarity(a, b):
    a = tf.math.l2_normalize(a, -1)
    b = tf.math.l2_normalize(b, -1)
    return tf.matmul(a, b, transpose_b=True)          # (B, B)

def info_nce_loss(movie_z, person_z, temperature: float = 0.07):
    logits = cosine_similarity(movie_z, person_z) / temperature     # (B, B)
    labels = tf.range(tf.shape(logits)[0])
    loss_a = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    loss_b = tf.keras.losses.sparse_categorical_crossentropy(labels, tf.transpose(logits), from_logits=True)
    return (loss_a + loss_b) * 0.5                                   # (B,)

def sample_random_person(conn: sqlite3.Connection, tconst: str) -> Dict[str, Any] | None:
    q = """
        SELECT p.primaryName, p.birthYear, p.deathYear,
               GROUP_CONCAT(pp.profession, ',') AS professions
        FROM people p
        LEFT JOIN people_professions pp ON p.nconst = pp.nconst
        INNER JOIN principals pr ON pr.nconst = p.nconst
        WHERE pr.tconst = ? AND p.birthYear IS NOT NULL
        GROUP BY p.nconst
        HAVING COUNT(pp.profession) > 0
        ORDER BY RANDOM()
        LIMIT 1
    """
    r = conn.execute(q, (tconst,)).fetchone()
    if not r: return None
    return {
        "primaryName":  r[0],
        "birthYear":    r[1],
        "deathYear":    r[2],
        "professions":  r[3].split(',') if r[3] else None,
    }

##########################################################################
# dataset generator ‑‑ yields (movie_inputs, person_inputs)
##########################################################################
# scripts/train_joint_autoencoder.py

def make_pair_generator(
        db_path: Path,
        movie_ae: TitlesAutoencoder,
        person_ae: PeopleAutoencoder,
        limit: int = 100_000,
        log_every: int = 10_000,
):
    import logging, sqlite3

    conn = sqlite3.connect(db_path, check_same_thread=False)
    cur = conn.cursor()

    count_sql = """
    WITH filtered_movies AS (
        SELECT t.tconst
        FROM titles t
        INNER JOIN title_genres g ON g.tconst = t.tconst
        WHERE
            t.startYear IS NOT NULL
            AND t.averageRating IS NOT NULL
            AND t.runtimeMinutes IS NOT NULL
            AND t.runtimeMinutes >= 5
            AND t.startYear >= 1850
            AND t.titleType IN ('movie','tvSeries','tvMovie','tvMiniSeries')
            AND t.numVotes >= 10
        GROUP BY t.tconst
        HAVING COUNT(g.genre) > 0
        LIMIT ?
    ),
    filtered_people AS (
        SELECT p.nconst
        FROM people p
        INNER JOIN people_professions pp ON pp.nconst = p.nconst
        WHERE p.birthYear IS NOT NULL
        GROUP BY p.nconst
        HAVING COUNT(pp.profession) > 0
    )
    SELECT COUNT(*)
    FROM principals pr
    JOIN filtered_movies fm ON fm.tconst = pr.tconst
    JOIN filtered_people fp ON fp.nconst = pr.nconst
    """

    logging.info(f"> joint generator: counting pairs in {db_path}, this may take a while…")
    total_pairs = cur.execute(count_sql, (limit,)).fetchone()[0]
    logging.info(f"> joint generator: epoch size = {total_pairs:,} pairs")

    movies = list(movie_ae.row_generator())[:limit]
    produced = 0

    person_sql = """
    SELECT
        p.primaryName,
        p.birthYear,
        p.deathYear,
        GROUP_CONCAT(pp.profession, ',')
    FROM people p
    LEFT JOIN people_professions pp ON pp.nconst = p.nconst
    INNER JOIN principals pr ON pr.nconst = p.nconst
    WHERE pr.tconst = ?
      AND p.birthYear IS NOT NULL
    GROUP BY p.nconst
    HAVING COUNT(pp.profession) > 0
    """

    for movie_dict in movies:
        people_rows = cur.execute(person_sql, (movie_dict["tconst"],)).fetchall()
        for row in people_rows:
            person_dict = {
                "primaryName": row[0],
                "birthYear":   row[1],
                "deathYear":   row[2],
                "professions": row[3].split(',') if row[3] else None,
            }
            movie_inputs = tuple(
                f.transform(movie_dict.get(f.name)) for f in movie_ae.fields
            )
            person_inputs = tuple(
                f.transform(person_dict.get(f.name)) for f in person_ae.fields
            )
            yield movie_inputs, person_inputs
            produced += 1
            if produced % log_every == 0:
                logging.info(f"joint generator: yielded {produced:,}/{total_pairs:,}")



##########################################################################
# custom model that bundles everything & runs its own train_step
##########################################################################
class JointAutoencoder(tf.keras.Model):
    """
    Joint autoencoder wrapping a TitlesAutoencoder and a PeopleAutoencoder.
    Computes reconstruction losses for both, plus an InfoNCE contrastive loss
    between their latent vectors.
    """
    def __init__(
        self,
        movie_ae: TitlesAutoencoder,
        person_ae: PeopleAutoencoder,
        temperature: float = 0.07
    ):
        super().__init__()
        self.movie_ae  = movie_ae
        self.person_ae = person_ae
        self.temp      = temperature

        # encoder shortcuts
        self.mov_enc = movie_ae.encoder
        self.per_enc = person_ae.encoder

        # losses & weights from each row-AE
        self.movie_losses   = movie_ae.get_loss_dict()
        self.movie_weights  = movie_ae.get_loss_weights_dict()
        self.person_losses  = person_ae.get_loss_dict()
        self.person_weights = person_ae.get_loss_weights_dict()

    def call(self, inputs, training=False):
        # inputs is a tuple: (movie_inputs, person_inputs)
        movie_in, person_in = inputs

        # encode
        m_z = self.mov_enc(movie_in,  training=training)   # (B, D)
        p_z = self.per_enc(person_in, training=training)   # (B, D)

        # decode via the row-AE decoders directly
        m_recon = self.movie_ae.decoder(m_z,      training=training)
        p_recon = self.person_ae.decoder(p_z,     training=training)

        return m_z, p_z, m_recon, p_recon

    def train_step(self, data):
        movie_in, person_in = data

        with tf.GradientTape() as tape:
            m_z, p_z, m_rec, p_rec = self((movie_in, person_in), training=True)

            field_losses = {}

            rec_loss = 0.0
            for idx, field in enumerate(self.movie_ae.fields):
                l = tf.reduce_mean(field.loss(movie_in[idx], m_rec[idx])) * field.weight
                rec_loss += l
                field_losses[f"movie_{field.name}_loss"] = l

            for idx, field in enumerate(self.person_ae.fields):
                l = tf.reduce_mean(field.loss(person_in[idx], p_rec[idx])) * field.weight
                rec_loss += l
                field_losses[f"person_{field.name}_loss"] = l

            nce   = tf.reduce_mean(info_nce_loss(m_z, p_z, self.temp))
            total = rec_loss + nce

        grads = tape.gradient(total, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "loss"      : total,
            "rec_loss"  : rec_loss,
            "nce"       : nce,
            **field_losses,
        }



##########################################################################
# assemble everything & run
##########################################################################
def build_joint_trainer(
    cfg: Dict[str, Any],
    warm: bool,
    db_path: Path,
    model_dir: Path
) -> tuple[JointAutoencoder, tf.data.Dataset]:
    """
    Returns:
      joint_model: JointAutoencoder wrapping TitlesAE & PeopleAE
      ds:          tf.data.Dataset yielding (movie_inputs, person_inputs)
    """
    # 1) Instantiate row-autoencoders
    movie_ae  = TitlesAutoencoder(cfg, model_dir / "TitlesAutoencoder")
    people_ae = PeopleAutoencoder(cfg, model_dir / "PeopleAutoencoder")

    # 2) Either load pretrained or build from scratch
    if warm:
        print("[+] Warming up: loading pretrained row autoencoders…")
        movie_ae.load_model()
        people_ae.load_model()
    else:
        print("[+] Cold start: accumulating stats for both AEs…")
        movie_ae.accumulate_stats()
        movie_ae.finalize_stats()
        people_ae.accumulate_stats()
        people_ae.finalize_stats()
        print("[+] Building both autoencoders…")
        movie_ae.build_autoencoder()
        people_ae.build_autoencoder()

    # 4) Build the tf.data pipeline
    movie_specs  = tuple(
        tf.TensorSpec(shape=f.input_shape, dtype=f.input_dtype)
        for f in movie_ae.fields
    )
    person_specs = tuple(
        tf.TensorSpec(shape=f.input_shape, dtype=f.input_dtype)
        for f in people_ae.fields
    )

    ds = tf.data.Dataset.from_generator(
        lambda: make_pair_generator(db_path, movie_ae, people_ae),
        output_signature=(movie_specs, person_specs)
    )
    ds = ds.batch(cfg["batch_size"]).prefetch(tf.data.AUTOTUNE)

    # 5) Wrap into JointAutoencoder
    joint = JointAutoencoder(movie_ae, people_ae, temperature=0.07)
    return joint, ds


##########################################################################
# CLI & main
##########################################################################
def main():
    parser = argparse.ArgumentParser(description="Train movie<‑>people joint embedding autoencoder")
    parser.add_argument("--warm", action="store_true", help="load pre‑trained row autoencoders")
    parser.add_argument("--epochs", type=int, default=None, help="override epochs in config")
    args = parser.parse_args()

    cfg = dict(project_config["autoencoder"])                       # shallow copy
    if args.epochs: cfg["epochs"] = args.epochs

    data_dir  = Path(project_config["data_dir"])
    db_path   = data_dir / "imdb.db"
    model_dir = Path(project_config["model_dir"])
    model_dir.mkdir(exist_ok=True, parents=True)

    joint_model, ds = build_joint_trainer(cfg, args.warm, db_path, model_dir)

    # simple optimizer – tweak as you like
    joint_model.compile(optimizer=tf.keras.optimizers.AdamW(
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"]))

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_dir / "JointMoviePersonAE_epoch_{epoch:02d}.keras",
        save_weights_only=False, save_freq="epoch")
    
    recon_cb = JointReconstructionCallback(
        movie_ae  = joint_model.movie_ae,
        person_ae = joint_model.person_ae,
        db_path   = project_config["autoencoder"]["db_path"],
        interval_batches = 100,
        num_samples      = 4,
    )

    tensorboard_callback = TensorBoardPerBatchLoggingCallback(log_dir=model_dir / "logs" / "joint", log_interval=20)

    joint_model.fit(ds,
                    epochs=cfg["epochs"],
                    callbacks=[ckpt, recon_cb, tensorboard_callback],)

    # save components
    print("[✓] Training done. Saving models…")
    model_dir.mkdir(exist_ok=True, parents=True)
    joint_model.save(model_dir / "JointMoviePersonAE_final.keras")
    joint_model.movie_ae.save_model()
    joint_model.person_ae.save_model()
    print("[✓] All models saved to", model_dir)

if __name__ == "__main__":
    main()
