from functools import partial
import tensorflow as tf
import sqlite3
import random
import numpy as np
import os
from pathlib import Path

from config import project_config
from autoencoder.imdb_row_autoencoders import TitlesAutoencoder, PeopleAutoencoder

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

    title_ae = TitlesAutoencoder(config, db_path)
    
    people_ae = PeopleAutoencoder(config, db_path)
    people_ae.fit()

if __name__ == "__main__":
    main()