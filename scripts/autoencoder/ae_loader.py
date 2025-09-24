from pathlib import Path
import torch
from config import ProjectConfig
from .imdb_row_autoencoders import TitlesAutoencoder, PeopleAutoencoder

def _infer_latent_dim_from_state_dict(sd: dict) -> int:
    w = sd.get("field_embed")
    if isinstance(w, torch.Tensor) and w.dim() == 2:
        return int(w.size(1))
    for k, v in sd.items():
        if k.endswith("field_embed") and isinstance(v, torch.Tensor) and v.dim() == 2:
            return int(v.size(1))
    raise RuntimeError("could not infer latent_dim from state_dict")

def load_autoencoders(config: ProjectConfig, warm_start: bool, freeze_loaded: bool) -> tuple[TitlesAutoencoder, PeopleAutoencoder]:
    mov = TitlesAutoencoder(config)
    per = PeopleAutoencoder(config)
    mov.accumulate_stats()
    mov.finalize_stats()
    per.accumulate_stats()
    per.finalize_stats()
    mov.build_autoencoder()
    per.build_autoencoder()

    if not warm_start:
        return mov, per

    model_dir = Path(config.model_dir)
    me = model_dir / "TitlesAutoencoder_encoder.pt"
    md = model_dir / "TitlesAutoencoder_decoder.pt"
    pe = model_dir / "PeopleAutoencoder_encoder.pt"
    pd = model_dir / "PeopleAutoencoder_decoder.pt"

    if not (me.exists() and md.exists() and pe.exists() and pd.exists()):
        return mov, per

    me_sd = torch.load(me, map_location="cpu")
    pe_sd = torch.load(pe, map_location="cpu")
    mov_dim = _infer_latent_dim_from_state_dict(me_sd)
    per_dim = _infer_latent_dim_from_state_dict(pe_sd)
    if mov_dim != per_dim:
        raise RuntimeError(f"latent_dim mismatch between checkpoints: movie={mov_dim} people={per_dim}")

    config.latent_dim = int(mov_dim)
    mov = TitlesAutoencoder(config)
    per = PeopleAutoencoder(config)
    mov.accumulate_stats()
    mov.finalize_stats()
    per.accumulate_stats()
    per.finalize_stats()
    mov.build_autoencoder()
    per.build_autoencoder()

    mov.encoder.load_state_dict(me_sd, strict=True)
    mov.decoder.load_state_dict(torch.load(md, map_location="cpu"), strict=True)
    per.encoder.load_state_dict(pe_sd, strict=True)
    per.decoder.load_state_dict(torch.load(pd, map_location="cpu"), strict=True)

    if freeze_loaded:
        for p in mov.encoder.parameters():
            p.requires_grad = False
        for p in mov.decoder.parameters():
            p.requires_grad = False
        for p in per.encoder.parameters():
            p.requires_grad = False
        for p in per.decoder.parameters():
            p.requires_grad = False

    return mov, per
