# scripts/autoencoder/ae_loader.py
from pathlib import Path
import torch
from .imdb_row_autoencoders import TitlesAutoencoder, PeopleAutoencoder

def _load_frozen_autoencoders(config):
    mov = TitlesAutoencoder(config)
    per = PeopleAutoencoder(config)
    mov.accumulate_stats()
    mov.finalize_stats()
    per.accumulate_stats()
    per.finalize_stats()
    mov.build_autoencoder()
    per.build_autoencoder()
    model_dir = Path(config["model_dir"])
    mov.encoder.load_state_dict(torch.load(model_dir / "TitlesAutoencoder_encoder.pt", map_location="cpu"))
    per.decoder.load_state_dict(torch.load(model_dir / "PeopleAutoencoder_decoder.pt", map_location="cpu"))
    per.encoder.load_state_dict(torch.load(model_dir / "PeopleAutoencoder_encoder.pt", map_location="cpu"))
    for p in mov.encoder.parameters():
        p.requires_grad = False
    for p in per.decoder.parameters():
        p.requires_grad = False
    for p in per.encoder.parameters():
        p.requires_grad = False
    mov.encoder.eval()
    per.decoder.eval()
    per.encoder.eval()
    return mov, per
