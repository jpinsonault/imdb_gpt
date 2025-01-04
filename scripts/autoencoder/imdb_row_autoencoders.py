from functools import cached_property
from typing import Any, Dict, List
from pathlib import Path
import sqlite3
import tensorflow as tf
from contextlib import redirect_stdout
from tqdm import tqdm
from prettytable import PrettyTable
from autoencoder.schema import RowAutoencoder
from autoencoder.fields import (
    TextField,
    ScalarField,
    MultiCategoryField,
    Scaling,
    BaseField
)
from autoencoder.training_callbacks import ReconstructionCallback
import os

class TitlesAutoencoder(RowAutoencoder):
    def __init__(self, config: Dict[str, Any], db_path: Path):
        super().__init__()
        self.config = config
        self.db_path = db_path
        self.model = None
        self.stats_accumulated = False

    @cached_property
    def build_fields(self) -> List[BaseField]:
        return [
            MultiCategoryField("titleType"),
            TextField("primaryTitle"),
            ScalarField("startYear", scaling=Scaling.STANDARDIZE),
            # ScalarField("endYear", scaling=Scaling.STANDARDIZE, optional=True),
            ScalarField("runtimeMinutes", scaling=Scaling.LOG),
            ScalarField("averageRating", scaling=Scaling.STANDARDIZE),
            ScalarField("numVotes", scaling=Scaling.LOG),
            MultiCategoryField("genres")
        ]

    def row_generator(self, db_path: Path):
        with sqlite3.connect(db_path, check_same_thread=False) as conn:
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
                GROUP BY t.tconst
            """)
            for row in c:
                (
                    titleType,
                    primaryTitle,
                    startYear,
                    endYear,
                    runtimeMinutes,
                    averageRating,
                    numVotes,
                    genres_str
                ) = row

                yield {
                    "titleType": titleType,
                    "primaryTitle": primaryTitle,
                    "startYear": startYear,
                    "endYear": endYear,
                    "runtimeMinutes": runtimeMinutes,
                    "averageRating": averageRating,
                    "numVotes": numVotes,
                    "genres": genres_str.split(',') if genres_str else []
                }
    
    def accumulate_stats(self, db_path: Path):
        if self.stats_accumulated:
            print("stats already accumulated")
            return

        print("Accumulating stats...")
        for row in tqdm(self.row_generator(db_path), desc="Accumulating stats"):
            self.accumulate_stats_for_row(row)

        self.finalize_stats()
        self.print_stats()
        self.stats_accumulated = True

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
             for row_dict in self.row_generator(self.db_path):
                x = tuple(f.transform(row_dict.get(f.name)) for f in self.fields)
                yield x, x

        # Define the signature for tf.data
        specs_in = tuple(tf.TensorSpec(shape=f.input_shape, dtype=f.input_dtype) for f in self.fields)
        specs_out = tuple(tf.TensorSpec(shape=f.output_shape, dtype=tf.float32) for f in self.fields)

        ds = tf.data.Dataset.from_generator(db_generator, output_signature=(specs_in, specs_out))
        return ds.batch(self.config["batch_size"])

    def fit(self):
        if not self.stats_accumulated:
             self.accumulate_stats(self.db_path)
        
        self._build_model()
        self._print_model_architecture()

        # Write model summary to file (optional)
        with redirect_stdout(open('logs/model_summary.txt', 'w')):
            self._print_model_architecture()

        print(f"Loss dict: {self.get_loss_dict()}")

        learning_rate = 0.00005
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=self.get_loss_dict(),
        )

        # Callbacks
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
            num_samples=5
        )

        ds = self._build_dataset(self.db_path)

        # Train
        self.model.fit(
            ds,
            epochs=self.config["epochs"],
            callbacks=[tensorboard_callback, reconstruction_callback]
        )
        self.model.save("tabular_autoencoder.h5")
        print("\nTraining complete and model saved.")