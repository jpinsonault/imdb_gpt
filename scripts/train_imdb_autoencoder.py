from functools import partial
import tensorflow as tf
import sqlite3
import random
import numpy as np
import os
from pathlib import Path

from config import project_config
from autoencoder.imdb_row_autoencoders import TitlesAutoencoder
from scripts.autoencoder.schema import UncertaintyWeightedAutoencoder

def lr_schedule(total_epochs, epoch):
    initial_lr = 0.00005
    final_lr = 1e-5
    # Compute the decay factor per epoch
    decay_factor = (final_lr / initial_lr) ** (1 / total_epochs)
    return initial_lr * (decay_factor ** epoch)

def main():
    config = project_config["autoencoder"]
    data_dir = Path(project_config["data_dir"])
    db_path = data_dir / 'imdb.db'

    total_epochs = config["epochs"]

    row_ae = TitlesAutoencoder(config, db_path)
    row_ae.accumulate_stats()   # so the fields are ready

    un_model = UncertaintyWeightedAutoencoder(row_ae)
    un_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),  # initial LR
    )

    ds = row_ae._build_dataset(db_path)  # your existing dataset code

    # Create the LearningRateScheduler callback
    lr_callback = tf.keras.callbacks.LearningRateScheduler(partial(lr_schedule, total_epochs))

    un_model.fit(ds, epochs=total_epochs, callbacks=[lr_callback])

if __name__ == "__main__":
    main()