from pathlib import Path
import torch

from config import project_config
from scripts.autoencoder.imdb_row_autoencoders import TitlesAutoencoder, PeopleAutoencoder

def main():
    print("Num GPUs Available:", torch.cuda.device_count())

    people_ae = PeopleAutoencoder(project_config)
    people_ae.fit()
    people_ae.save_model()

    title_ae = TitlesAutoencoder(project_config)
    title_ae.fit()
    title_ae.save_model()

if __name__ == "__main__":
    main()
