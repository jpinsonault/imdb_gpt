from functools import partial
import tensorflow as tf
import sqlite3
import random
import numpy as np
import os
from pathlib import Path

from config import project_config
from autoencoder.imdb_row_autoencoders import TitlesAutoencoder, PeopleAutoencoder

def main():
    config = project_config["autoencoder"]
    data_dir = Path(project_config["data_dir"])
    db_path = data_dir / 'imdb.db'
    model_dir = Path(config["model_dir"])


    title_ae = TitlesAutoencoder(config, db_path)
    title_ae.load_model(model_dir)
    title_ae.fit()
    title_ae.save_model(model_dir)

    people_ae = PeopleAutoencoder(config, db_path)
    # people_ae.fit()

if __name__ == "__main__":
    main()