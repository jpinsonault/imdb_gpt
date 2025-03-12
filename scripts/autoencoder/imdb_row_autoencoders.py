from typing import Any, Dict, List
from pathlib import Path
import sqlite3
import tensorflow as tf
from contextlib import redirect_stdout
from scripts.autoencoder.row_autoencoder import RowAutoencoder
from autoencoder.training_callbacks import ModelSaveCallback, ReconstructionCallback
import os
from autoencoder.fields import (
    TextField,
    ScalarField,
    MultiCategoryField,
    Scaling,
    BaseField
)


class TitlesAutoencoder(RowAutoencoder):
    def __init__(self, config: Dict[str, Any], db_path: Path, model_dir: Path):
        super().__init__(config, model_dir)
        self.db_path = db_path
        self.model = None
        self.stats_accumulated = False

    def build_fields(self) -> List[BaseField]:
        return [
            # MultiCategoryField("titleType"),
            TextField("primaryTitle"),
            # ScalarField("startYear", scaling=Scaling.STANDARDIZE),
            # ScalarField("endYear", scaling=Scaling.STANDARDIZE, optional=True),
            # ScalarField("runtimeMinutes", scaling=Scaling.LOG),
            # ScalarField("averageRating", scaling=Scaling.STANDARDIZE),
            # ScalarField("numVotes", scaling=Scaling.LOG),
            # MultiCategoryField("genres")
        ]

    def row_generator(self):
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT 
                    t.titleType,
                    t.primaryTitle,
                    t.startYear,
                    t.endYear,
                    t.runtimeMinutes,
                    t.averageRating,
                    t.numVotes,
                    GROUP_CONCAT(g.genre, ',') AS genres
                FROM titles t
                LEFT JOIN title_genres g ON t.tconst = g.tconst
                WHERE t.startYear IS NOT NULL 
                AND t.averageRating IS NOT NULL
                AND t.runtimeMinutes IS NOT NULL
                AND t.runtimeMinutes >= 5
                AND t.startYear >= 1850
                AND t.titleType IN ('movie', 'tvSeries', 'tvMovie', 'tvMiniSeries')
                GROUP BY t.tconst
                HAVING COUNT(g.genre) > 0
                AND t.numVotes >= 10
            """)
            for row in c:
                yield {
                    "titleType": row[0],
                    "primaryTitle": row[1],
                    "startYear": row[2],
                    "endYear": row[3],
                    "runtimeMinutes": row[4],
                    "averageRating": row[5],
                    "numVotes": row[6],
                    "genres": row[7].split(',') if row[7] else []
                }

    def _print_model_architecture(self):
        print("\n--- Field Encoder/Decoder Summaries ---")
        for field in self.fields:
            encoder = field.build_encoder(latent_dim=self.config["latent_dim"])
            print(f"\nEncoder: {field.name}")
            encoder.summary()

            decoder = field.build_decoder(latent_dim=self.config["latent_dim"])
            print(f"\nDecoder: {field.name}")
            decoder.summary()

        print("--- Main Autoencoder Summary ---")
        self.model.summary()
        print("Model Outputs:", self.model.output_names)

    def fit(self):
        self.accumulate_stats()

        if self.model is None:
            self._build_model()
        
        self._print_model_architecture()

        with redirect_stdout(open('logs/model_summary.txt', 'w')):
            self._print_model_architecture()

        schedule = [
            0.001,
            0.0005,
            0.0002, 0.0002,
            0.0001, 0.0001,
        ]
        
        def fixed_scheduler(epoch, lr):
            return schedule[epoch] if epoch < len(schedule) else schedule[-1]
        lr_callback = tf.keras.callbacks.LearningRateScheduler(fixed_scheduler)

        optimizer = tf.keras.optimizers.AdamW(learning_rate=schedule[0], weight_decay=1e-6)

        self.model.compile(
            optimizer=optimizer,
            loss=self.get_loss_dict(),
            loss_weights=self.get_loss_weights_dict()
        )

        log_dir = "logs/fit/" + str(int(tf.timestamp().numpy()))
        os.makedirs(log_dir, exist_ok=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            embeddings_freq=1,
            update_freq=self.config["callback_interval"]
        )
        reconstruction_callback = ReconstructionCallback(
            interval_batches=self.config["callback_interval"],
            row_autoencoder=self,
            db_path=self.db_path,
            num_samples=20
        )

        model_save_callback = ModelSaveCallback(self, output_dir=self.model_dir)

        ds = self._build_dataset()

        self.model.fit(
            ds,
            epochs=self.config["epochs"],
            callbacks=[tensorboard_callback, reconstruction_callback, lr_callback, model_save_callback]
        )

        print("\nTraining complete and model saved.")


class PeopleAutoencoder(RowAutoencoder):
    def __init__(self, config: Dict[str, Any], db_path: Path):
        super().__init__()
        self.config = config
        self.db_path = db_path  
        self.model = None
        self.stats_accumulated = False

    def build_fields(self) -> List[BaseField]:
        return [
            TextField("primaryName"),
            ScalarField("birthYear", scaling=Scaling.STANDARDIZE),
            # ScalarField("deathYear", scaling=Scaling.STANDARDIZE, optional=True), # optional as some people are still alive
            # MultiCategoryField("professions", optional=True) # optional as some might have no profession listed
        ]

    def row_generator(self):
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT
                    p.primaryName,
                    p.birthYear,
                    p.deathYear,
                    GROUP_CONCAT(pp.profession, ',') AS professions
                FROM people p
                LEFT JOIN people_professions pp ON p.nconst = pp.nconst
                WHERE p.birthYear IS NOT NULL  -- Filter out people without birth years
                GROUP BY p.nconst
            """)
            for row in c:
                yield {
                    "primaryName": row[0],
                    "birthYear": row[1],
                    "deathYear": row[2],
                    "professions": row[3].split(',') if row[3] else []
                }

    def _print_model_architecture(self):
        print("\n--- Field Encoder/Decoder Summaries ---")
        for field in self.fields:
            encoder = field.build_encoder(latent_dim=self.config["latent_dim"])
            print(f"\nEncoder: {field.name}")
            encoder.summary()

            decoder = field.build_decoder(latent_dim=self.config["latent_dim"])
            print(f"\nDecoder: {field.name}")
            decoder.summary()

        print("--- Main Autoencoder Summary ---")
        self.model.summary()
        print("Model Outputs:", self.model.output_names)

    def _build_dataset(self, db_path: Path) -> tf.data.Dataset:
        def db_generator():
             for row_dict in self.row_generator():
                x = tuple(f.transform(row_dict.get(f.name)) for f in self.fields)
                yield x, x

        # Define the signature for tf.data
        specs_in = tuple(tf.TensorSpec(shape=f.input_shape, dtype=f.input_dtype) for f in self.fields)
        specs_out = tuple(tf.TensorSpec(shape=f.output_shape, dtype=tf.float32) for f in self.fields)

        ds = tf.data.Dataset.from_generator(db_generator, output_signature=(specs_in, specs_out))
        return ds.batch(self.config["batch_size"])

    def fit(self):
        self.accumulate_stats()

        self._build_model()
        self._print_model_architecture()

        # Write model summary to file (optional)
        with redirect_stdout(open('logs/people_model_summary.txt', 'w')):
            self._print_model_architecture()

        learning_rate = 0.00005
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=self.get_loss_dict(),
            loss_weights=self.get_loss_weights_dict()
        )

        # Callbacks
        log_dir = "logs/people_fit/" + str(int(tf.timestamp().numpy()))
        os.makedirs(log_dir, exist_ok=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            embeddings_freq=1,
            update_freq=self.config["callback_interval"]
        )
        reconstruction_callback = ReconstructionCallback(
            interval_batches=self.config["callback_interval"],
            row_autoencoder=self,
            db_path=self.db_path,
            num_samples=20,
        )

        ds = self._build_dataset(self.db_path)

        # Train
        self.model.fit(
            ds,
            epochs=self.config["epochs"],
            callbacks=[tensorboard_callback, reconstruction_callback]
        )
        self.model.save("people_autoencoder.h5")
        print("\nTraining complete and people autoencoder model saved.")