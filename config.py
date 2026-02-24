# config.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

@dataclass
class ProjectConfig:

    # -- Paths --
    data_dir: str = "data"
    raw_db_path: str = "data/raw.db"
    db_path: str = "data/imdb.db"
    model_dir: str = "models"
    tensorboard_dir: str = "runs"

    # -- Standalone autoencoder training (RowAutoencoder.fit) --
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 4
    latent_dim: int = 64
    log_interval: int = 50
    save_interval: int = 500
    movie_limit: int = 100000000000  # SQL LIMIT; effectively "no limit"

    # -- LR schedule (shared by both training scripts) --
    lr_schedule: str = "cosine"
    lr_warmup_steps: int = 2000
    lr_warmup_ratio: float = 0.0
    lr_min_factor: float = 0.05

    # -- Hybrid set training (train_simple_set.py) --
    hybrid_set_epochs: int = 1000
    hybrid_set_model_lr: float = 1e-3
    hybrid_set_emb_lr: float = 1e-3
    hybrid_set_weight_decay: float = 0.0
    hybrid_set_save_interval: int = 500
    hybrid_set_recon_interval: int = 200

    # -- Hybrid set model architecture --
    hybrid_set_movie_dim: int = 128
    hybrid_set_hidden_dim: int = 2048
    hybrid_set_person_dim: int = 128
    hybrid_set_dropout: float = 0.1
    hybrid_set_logit_scale: float = 20.0  # initial value; learned during training
    hybrid_set_film_bottleneck_dim: int = 128
    hybrid_set_noise_std: float = 0.0  # Gaussian noise for info bottleneck (0 = off; try 0.05-0.2)

    # -- Hybrid set decoder (TransformerFieldDecoder) --
    hybrid_set_decoder_num_layers: int = 3
    hybrid_set_decoder_num_heads: int = 4
    hybrid_set_decoder_ff_multiplier: int = 4
    hybrid_set_decoder_dropout: float = 0.1
    hybrid_set_decoder_norm_first: bool = True

    # -- Hybrid set loss weights --
    hybrid_set_w_bce: float = 1.0       # weight for set prediction (BCE) loss
    hybrid_set_w_recon: float = 0.5     # weight for field reconstruction loss
    hybrid_set_w_search_encoder: float = 0.1
    hybrid_set_film_reg: float = 1e-3   # FiLM regularization weight
    hybrid_set_focal_alpha: float = 0.25
    hybrid_set_focal_gamma: float = 2.0

    # -- Head configuration (edge type → loss weight, category → head mapping) --
    hybrid_set_category_to_head: Dict[str, str] = field(default_factory=lambda: {
        "actor": "cast",
        "actress": "cast",
        "self": "cast",
        "director": "director",
        "writer": "writer",
        "producer": "producer",
        "cinematographer": "cinematographer",
        "composer": "composer",
        "editor": "editor",
        "production_designer": "production_designer",
        "casting_director": "casting_director",
    })

    hybrid_set_heads: Dict[str, float] = field(default_factory=lambda: {
        "cast": 1.0,
        "director": 0.5,
        "writer": 0.5,
        "producer": 0.3,
        "cinematographer": 0.3,
        "composer": 0.3,
        "editor": 0.3,
        "production_designer": 0.2,
        "casting_director": 0.2,
    })

    # -- Data filtering --
    hybrid_set_min_person_frequency: int = 3  # min movie appearances to include a person

    # -- Reconstruction display --
    hybrid_set_recon_num_samples: int = 3
    hybrid_set_recon_table_width: int = 80
    hybrid_set_recon_threshold: float = 0.5

    # -- Caching --
    use_cache: bool = True
    refresh_cache: bool = False


project_config = ProjectConfig()


def ensure_dirs(cfg: ProjectConfig):
    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.tensorboard_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)
