import tensorflow as tf
import sqlite3
import random
import numpy as np
import os
from pathlib import Path

from config import project_config
from autoencoder.imdb_row_autoencoders import TitlesAutoencoder
from scripts.autoencoder.schema import UncertaintyWeightedAutoencoder

def main():
    config = project_config["autoencoder"]
    data_dir = Path(project_config["data_dir"])
    db_path = data_dir / 'imdb.db'

    row_ae = TitlesAutoencoder(config, db_path)
    row_ae.accumulate_stats()   # so the fields are ready

    un_model = UncertaintyWeightedAutoencoder(row_ae)

    initial_learning_rate = 0.0005
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=2000,
        decay_rate=0.5,
        staircase=False
    )

    un_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    )

    ds = row_ae._build_dataset(db_path)  # e.g. from your existing code
    un_model.fit(ds, epochs=config["epochs"])

if __name__ == "__main__":
    main()
