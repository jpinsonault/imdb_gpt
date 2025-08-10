import logging
import torch

from config import project_config
from autoencoder.seq.model import MoviesToPeopleSequenceDecoder

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    print("Num GPUs Available:", torch.cuda.device_count())
    model = MoviesToPeopleSequenceDecoder(project_config)
    model.fit()

if __name__ == "__main__":
    main()
