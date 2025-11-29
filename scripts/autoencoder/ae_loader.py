# scripts/autoencoder/ae_loader.py

import logging
from pathlib import Path

import torch

from config import ProjectConfig, project_config
from .imdb_row_autoencoders import TitlesAutoencoder, PeopleAutoencoder
from .fields import TextField, NumericDigitCategoryField
from scripts.train_joint_autoencoder import JointAutoencoder


class AutoencoderLoadError(RuntimeError):
    pass


def _build_joint_style_autoencoders(cfg: ProjectConfig):
    """
    Rebuild TitlesAutoencoder & PeopleAutoencoder *exactly* in the way
    train_joint_autoencoder.py does before constructing JointAutoencoder:

        - fresh config
        - accumulate_stats() on both
        - finalize_stats() on both
        - build_autoencoder() on both
    """
    mov_ae = TitlesAutoencoder(cfg)
    per_ae = PeopleAutoencoder(cfg)

    # Same order as build_joint_trainer
    mov_ae.accumulate_stats()
    per_ae.accumulate_stats()

    mov_ae.finalize_stats()
    per_ae.finalize_stats()

    mov_ae.build_autoencoder()
    per_ae.build_autoencoder()

    return mov_ae, per_ae


def _freeze_autoencoder(ae, device: torch.device):
    ae.encoder.to(device).eval()
    ae.decoder.to(device).eval()
    for p in ae.encoder.parameters():
        p.requires_grad_(False)
    for p in ae.decoder.parameters():
        p.requires_grad_(False)


def _load_base_autoencoders(cfg: ProjectConfig | None = None, freeze: bool = True):
    """
    Shared logic to load AEs from disk.
    """
    if cfg is None:
        cfg = project_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(cfg.model_dir) / "JointMoviePersonAE_final.pt"
    if not ckpt_path.exists():
        raise AutoencoderLoadError(
            f"[loader] joint checkpoint not found at {ckpt_path}. "
            f"Run scripts.train_joint_autoencoder first."
        )

    # Rebuild AEs in the exact "joint-training" style
    try:
        mov_ae, per_ae = _build_joint_style_autoencoders(cfg)
    except Exception as exc:
        raise AutoencoderLoadError(
            f"[loader] failed to rebuild joint-style autoencoders: {exc}"
        ) from exc

    # Wrap in JointAutoencoder skeleton and load weights
    joint = JointAutoencoder(mov_ae, per_ae)
    try:
        state = torch.load(ckpt_path, map_location=device)
        joint.load_state_dict(state, strict=True)
    except Exception as exc:
        raise AutoencoderLoadError(
            f"[loader] failed to load joint weights from {ckpt_path}: {exc}"
        ) from exc
    
    # Unwrap is implicit because mov_ae.encoder is now the TypedEncoder from the joint model
    
    if freeze:
        _freeze_autoencoder(mov_ae, device)
        _freeze_autoencoder(per_ae, device)
        logging.info("[loader] loaded frozen joint autoencoders")
    else:
        # Move wrappers explicitly to device. 
        # Do NOT use .model.to(device) here, as .model references the unwrapped encoder.
        mov_ae.encoder.to(device).train()
        mov_ae.decoder.to(device).train()
        
        per_ae.encoder.to(device).train()
        per_ae.decoder.to(device).train()
        
        logging.info("[loader] loaded TRAINABLE joint autoencoders (fine-tuning mode)")

    return mov_ae, per_ae


def _load_frozen_autoencoders(cfg: ProjectConfig | None = None):
    return _load_base_autoencoders(cfg, freeze=True)


def _load_trainable_autoencoders(cfg: ProjectConfig | None = None):
    return _load_base_autoencoders(cfg, freeze=False)