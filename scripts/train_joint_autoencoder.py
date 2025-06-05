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

from prettytable import PrettyTable
import tensorflow as tf
from tensorflow.keras import layers as KL
from tensorflow.keras import ops
from tensorflow.keras.callbacks import Callback

from config import project_config
from scripts.autoencoder.edge_loss_logger import EdgeLossLogger
from scripts.autoencoder.fields import TextField
from scripts.autoencoder.imdb_row_autoencoders import TitlesAutoencoder, PeopleAutoencoder
from scripts.autoencoder.joint_edge_sampler import make_edge_sampler
from scripts.autoencoder.training_callbacks import JointReconstructionCallback, TensorBoardPerBatchLoggingCallback
logging.basicConfig(level=logging.INFO)

##########################################################################
# helpers
##########################################################################
def cosine_similarity(a, b):
    a = tf.math.l2_normalize(a, -1)
    b = tf.math.l2_normalize(b, -1)
    return tf.matmul(a, b, transpose_b=True)          # (B, B)

def info_nce_loss(movie_z, person_z, temperature: float):
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
    movie_count = 0
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
            if movie_count < 10:
                logging.info(f"movie: {movie_dict['tconst']}")
                logging.info(f"  {movie_dict}")
                logging.info(f"person: {person_dict['primaryName']}")
                logging.info(f"  {person_dict}")
            yield movie_inputs, person_inputs
        movie_count += 1



##########################################################################
# custom model that bundles everything & runs its own train_step
##########################################################################
class JointAutoencoder(tf.keras.Model):
    """
    Joint autoencoder that logs per-edge reconstruction losses only.
    """
    def __init__(
        self,
        movie_ae,
        person_ae,
        temperature: float,
        loss_logger: EdgeLossLogger,
    ):
        super().__init__()
        self.movie_ae = movie_ae
        self.person_ae = person_ae
        self.temp = temperature
        self.logger = loss_logger

        self.mov_enc = movie_ae.encoder
        self.per_enc = person_ae.encoder

        self.movie_losses = movie_ae.get_loss_dict()
        self.movie_weights = movie_ae.get_loss_weights_dict()
        self.person_losses = person_ae.get_loss_dict()
        self.person_weights = person_ae.get_loss_weights_dict()

        self._batch = 0
        self._epoch = 0

    def train_step(self, data):
        movie_in, person_in, edge_ids = data
        with tf.GradientTape() as tape:
            m_z, p_z, m_rec, p_rec = self((movie_in, person_in), training=True)

            field_losses = {}
            rec_loss = 0.0

            for idx, field in enumerate(self.movie_ae.fields):
                l = tf.reduce_mean(field.loss(movie_in[idx], m_rec[idx])) * field.weight
                rec_loss += l
                field_losses[f"movie_{field.name}"] = l

            for idx, field in enumerate(self.person_ae.fields):
                l = tf.reduce_mean(field.loss(person_in[idx], p_rec[idx])) * field.weight
                rec_loss += l
                field_losses[f"person_{field.name}"] = l

            nce = tf.constant(0.0)
            total = rec_loss

        grads = tape.gradient(total, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        total_val = float(total.numpy())
        field_vals = {k: float(v.numpy()) for k, v in field_losses.items()}

        for eid in edge_ids.numpy().tolist():
            self.logger.add(int(eid), self._epoch, self._batch, total_val, field_vals)
        self._batch += 1

        return {
            "loss": total,
            "rec_loss": rec_loss,
            **field_losses,
        }

    def call(self, inputs, training=False):
        movie_in, person_in = inputs

        m_z = self.mov_enc(movie_in, training=training)
        p_z = self.per_enc(person_in, training=training)

        m_recon = self.movie_ae.decoder(m_z, training=training)
        p_recon = self.person_ae.decoder(p_z, training=training)

        return m_z, p_z, m_recon, p_recon

    def reset_metrics(self):
        super().reset_metrics()
        self._batch = 0
        self._epoch += 1



##########################################################################
# assemble everything & run
##########################################################################
class _FlushLogger(Callback):
    def __init__(self, logger: EdgeLossLogger):
        super().__init__()
        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        self.logger.flush()

    def on_train_end(self, logs=None):
        self.logger.close()


def build_joint_trainer(
    config: Dict[str, Any],
    warm: bool,
    db_path: Path,
    model_dir: Path,
) -> tuple[JointAutoencoder, tf.data.Dataset, EdgeLossLogger]:

    movie_ae = TitlesAutoencoder(config)
    people_ae = PeopleAutoencoder(config)

    if warm:
        raise NotImplementedError("Warm start is not implemented yet.")
        movie_ae.load_model()
        people_ae.load_model()
    else:
        movie_ae.accumulate_stats()
        movie_ae.finalize_stats()
        movie_ae.build_autoencoder()
        people_ae.accumulate_stats()
        people_ae.finalize_stats()
        people_ae.build_autoencoder()

    movie_ae.encoder.trainable = True
    movie_ae.decoder.trainable = True
    people_ae.encoder.trainable = True
    people_ae.decoder.trainable = True

    movie_specs = tuple(
        tf.TensorSpec(shape=f.input_shape, dtype=f.input_dtype) for f in movie_ae.fields
    )
    person_specs = tuple(
        tf.TensorSpec(shape=f.input_shape, dtype=f.input_dtype) for f in people_ae.fields
    )
    edge_spec = tf.TensorSpec(shape=(), dtype=tf.int64)

    edge_gen = make_edge_sampler(
        db_path       = db_path,
        movie_ae      = movie_ae,
        person_ae     = people_ae,
        batch_size    = config["batch_size"],
        refresh_batches = config['edge_sampler']["refresh_batches"],
        boost         = config['edge_sampler']["weak_edge_boost"],
    )

    ds = (
        tf.data.Dataset.from_generator(
            lambda: edge_gen,
            output_signature=(movie_specs, person_specs, edge_spec),
        )
        .batch(config["batch_size"])
        .prefetch(tf.data.AUTOTUNE)
    )

    loss_logger = EdgeLossLogger(db_path)

    joint = JointAutoencoder(
        movie_ae,
        people_ae,
        temperature=0.07,
        loss_logger=loss_logger,
    )
    return joint, ds, loss_logger




##########################################################################
# CLI & main
##########################################################################
def main():
    parser = argparse.ArgumentParser(description="Train movie<‑>people joint embedding autoencoder")
    parser.add_argument("--warm", action="store_true", help="load pre‑trained row autoencoders")
    parser.add_argument("--epochs", type=int, default=None, help="override epochs in config")
    args = parser.parse_args()

    if args.epochs: project_config["epochs"] = args.epochs

    data_dir  = Path(project_config["data_dir"])
    db_path   = data_dir / "imdb.db"
    model_dir = Path(project_config["model_dir"])
    model_dir.mkdir(exist_ok=True, parents=True)

    joint_model, ds, logger = build_joint_trainer(project_config, args.warm, db_path, model_dir)

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_dir / "JointMoviePersonAE_epoch_{epoch:02d}.keras",
        save_weights_only=False,
        monitor="loss",
        mode="min",
        save_best_only=False,
        save_freq="epoch",
    )

    tensorboard_cb = TensorBoardPerBatchLoggingCallback(
        log_dir=Path(project_config["log_dir"]) / "joint",
        log_interval=20,
    )

    recon_cb = JointReconstructionCallback(
        movie_ae=joint_model.movie_ae,
        person_ae=joint_model.person_ae,
        db_path=project_config["db_path"],
        interval_batches=5,
        num_samples=4,
    )

    joint_model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=project_config["learning_rate"],
            weight_decay=project_config["weight_decay"],
        ),
        run_eagerly=True,
    )

    # run *before* joint training starts
    for f in joint_model.movie_ae.fields:
        if str(type(f)) == "<class 'autoencoder.fields.TextField'>":
            print(f.name, f.tokenizer.get_vocab_size(), f.pad_token_id)

    for f in joint_model.person_ae.fields:
        if str(type(f)) == "<class 'autoencoder.fields.TextField'>":
            print(f.name, f.tokenizer.get_vocab_size(), f.pad_token_id)


    joint_model.fit(
        ds,
        epochs=project_config["epochs"],
        callbacks=[
            _FlushLogger(logger),
            tf.keras.callbacks.TerminateOnNaN(),
            ckpt,
            tensorboard_cb,
            recon_cb,
        ],
    )

    logger.close()
    joint_model.save(model_dir / "JointMoviePersonAE_final.keras")


    # save components
    print("[✓] Training done. Saving models…")
    model_dir.mkdir(exist_ok=True, parents=True)
    joint_model.save(model_dir / "JointMoviePersonAE_final.keras")
    joint_model.movie_ae.save_model()
    joint_model.person_ae.save_model()
    print("[✓] All models saved to", model_dir)

if __name__ == "__main__":
    main()
