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
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

    people_ae = PeopleAutoencoder(project_config)
    people_ae.fit()
    people_ae.save_model()

    title_ae = TitlesAutoencoder(project_config)
    title_ae.fit()
    title_ae.save_model()

if __name__ == "__main__":
    main()