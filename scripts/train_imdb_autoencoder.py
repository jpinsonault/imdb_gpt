import tensorflow as tf
import sqlite3
import random
import numpy as np
import os
from pathlib import Path

from config import project_config
from autoencoder.imdb_row_autoencoders import TitlesAutoencoder


def main():
    config = project_config["autoencoder"]
    data_dir = Path(project_config["data_dir"])
    db_path = data_dir / 'imdb.db'
    
    titles_autoencoder = TitlesAutoencoder(config, db_path)
    titles_autoencoder.fit()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        raise