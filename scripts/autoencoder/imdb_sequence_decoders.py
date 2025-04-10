from functools import cached_property
from typing import Any, Dict, List
from pathlib import Path
import sqlite3
import tensorflow as tf
from contextlib import redirect_stdout
from tqdm import tqdm
from prettytable import PrettyTable
from scripts.autoencoder.row_autoencoder import RowAutoencoder, TableJoinSequenceEncoder
from autoencoder.fields import (
    TextField,
    ScalarField,
    MultiCategoryField,
    Scaling,
    BaseField
)
from autoencoder.training_callbacks import ReconstructionCallback
import os

class TitlesToPrinciplesSequenceEncoder(TableJoinSequenceEncoder):
    def __init__(self, config: Dict[str, Any], db_path: Path):
        super().__init__(config, db_path, "titles", "principles")
        self.config = config
        self.db_path = db_path
        self.model = None
        self.stats_accumulated = False

    def row_generator(self, db_path: Path):
        """Fetch each id in the first table, and then yield it with the corresponding rows from the second table as (x, y)
        x is one row from the first table, and y is a list of rows from the second table"""
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
                WHERE t.startYear IS NOT NULL 
                AND t.averageRating IS NOT NULL
                AND t.runtimeMinutes IS NOT NULL
                AND t.runtimeMinutes >= 5
                AND t.startYear >= 1850
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
    
    def accumulate_stats(self, db_path: Path):
        if self.stats_accumulated:
            print("stats already accumulated")
            return

        self.from_table_schema.accumulate_stats()

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
        self.accumulate_stats(self.db_path)
        
        self.build_model()
        self._print_model_architecture()

        with redirect_stdout(open('logs/model_summary.txt', 'w')):
            self._print_model_architecture()

        print(f"Loss dict: {self.get_loss_dict()}")
        print(f"Loss weights dict: {self.get_loss_weights_dict()}")  # Added line

        learning_rate = 0.00005
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=self.get_loss_dict(),
            loss_weights=self.get_loss_weights_dict()  # Added loss weights
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
