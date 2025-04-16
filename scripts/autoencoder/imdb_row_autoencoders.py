from pathlib import Path
from typing import Any, Dict, List
import sqlite3
import os
from contextlib import redirect_stdout

import tensorflow as tf
from autoencoder.training_callbacks import ModelSaveCallback, ReconstructionCallback
from autoencoder.fields import (
    NumericDigitCategoryField,
    QuantileThresholdCategory,
    SingleCategoryField,
    TextField,
    ScalarField,
    MultiCategoryField,
    Scaling,
    BaseField
)

from scripts.autoencoder.row_autoencoder import RowAutoencoder

class TitlesAutoencoder(RowAutoencoder):
    def __init__(self, config: Dict[str, Any], db_path: Path, model_dir: Path):
        super().__init__(config, model_dir)
        self.db_path = db_path
        self.model = None
        self.stats_accumulated = False

    def build_fields(self) -> List[BaseField]:
        return [
            TextField("primaryTitle"),
            NumericDigitCategoryField("startYear", base=10),
            # ScalarField("endYear", scaling=Scaling.STANDARDIZE, optional=True),
            NumericDigitCategoryField("runtimeMinutes", base=10),
            NumericDigitCategoryField("averageRating", base=10, fraction_digits=1),
            NumericDigitCategoryField("numVotes", base=10),
            MultiCategoryField("genres")
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
                LIMIT 100000
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
            self.build_autoencoder()

        self._print_model_architecture()

        with redirect_stdout(open('logs/model_summary.txt', 'w')):
            self._print_model_architecture()

        schedule = [0.0002] * 200 + [0.00018]

        def fixed_scheduler(epoch, lr):
            return schedule[epoch] if epoch < len(schedule) else schedule[-1]
        lr_callback = tf.keras.callbacks.LearningRateScheduler(fixed_scheduler)

        optimizer = tf.keras.optimizers.AdamW(learning_rate=schedule[0], weight_decay=1e-3)

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
    def __init__(self, config: Dict[str, Any], db_path: Path, model_dir: Path):
        super().__init__(config, model_dir)
        self.db_path = db_path
        self.model = None
        self.encoder = None
        self.decoder = None
        self.stats_accumulated = False
        self.latent_dim = config["latent_dim"]

    def build_fields(self) -> List[BaseField]:
        return [
            TextField("primaryName"),
            NumericDigitCategoryField("birthYear"),
            # SingleCategoryField("deathYear", optional=True),
            MultiCategoryField("professions")
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
                WHERE p.birthYear IS NOT NULL
                GROUP BY p.nconst
                HAVING COUNT(pp.profession) > 0
            """)
            for row in c:
                yield {
                    "primaryName": row[0],
                    "birthYear": row[1],
                    "deathYear": row[2],
                    "professions": row[3].split(',') if row[3] else None
                }


    def fit(self):
        """Fits the standalone PeopleAutoencoder."""
        print(f"--- Training Standalone {self.__class__.__name__} ---")
        if not self.stats_accumulated:
            self.accumulate_stats()

        if self.model is None:
            self.build_autoencoder()

        print("\n--- People Autoencoder Architecture ---")
        print("--- Encoder Summary ---")
        self.encoder.summary()
        print("--- Decoder Summary ---")
        self.decoder.summary()
        print("--- Main Autoencoder Summary ---")
        self.model.summary()

        initial_lr = self.config.get("learning_rate", 0.0001)
        optimizer = tf.keras.optimizers.AdamW(learning_rate=initial_lr, weight_decay=1e-4)
        loss_dict = self.get_loss_dict()
        loss_weights_dict = self.get_loss_weights_dict()

        self.model.compile(
            optimizer=optimizer,
            loss=list(loss_dict.values()),
            loss_weights=list(loss_weights_dict.values())
        )

        log_dir = os.path.join(self.model_dir, "logs", f"{self.__class__.__name__}_fit", str(int(tf.timestamp().numpy())))
        os.makedirs(log_dir, exist_ok=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1, update_freq='epoch'
        )
        checkpoint_path = os.path.join(self.model_dir, f"{self.__class__.__name__}_epoch_{{epoch:02d}}.keras")
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
             filepath=checkpoint_path, save_weights_only=False, save_freq='epoch'
        )
        reconstruction_callback = ReconstructionCallback(
            interval_batches=self.config["callback_interval"],
            row_autoencoder=self,
            db_path=self.db_path,
            num_samples=20
        )

        ds = self._build_dataset()

        print("Starting standalone training...")
        self.model.fit(
            ds,
            epochs=self.config["epochs"],
            callbacks=[
                tensorboard_callback, 
                model_checkpoint_callback,
                reconstruction_callback
            ]
        )
        print("\nStandalone Training complete.")
        self.save_model()
