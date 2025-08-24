# scripts/train_imdb_sequence_decoder.py
import argparse
from config import project_config
from scripts.autoencoder.sequence_trainer import train_sequence_predictor

def main():
    parser = argparse.ArgumentParser(description="Train movie-to-people sequence decoder with latent supervision")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--movie-limit", type=int, default=0)
    args = parser.parse_args()
    train_sequence_predictor(project_config, steps=args.steps, save_every=args.save_every)

if __name__ == "__main__":
    main()
