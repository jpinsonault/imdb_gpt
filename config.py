# config.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict

@dataclass
class ProjectConfig:
    data_dir: str = "data"
    
    # --- Database Paths ---
    raw_db_path: str = "data/raw.db"    # For raw TSV imports
    db_path: str = "data/imdb.db"       # For normalized training tables & edges
    
    model_dir: str = "models"

    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    epochs: int = 4

    latent_dim: int = 64

    save_interval: int = 500
    flush_interval: int = 250
    callback_interval: int = 200
    tensorboard_dir: str = "runs"
    log_interval: int = 50

    nce_temp: float = 0.05
    nce_weight: float = 2.0

    latent_type_loss_weight: float = 0.01

    movie_limit: int = 100000000000

    num_workers: int = 4
    prefetch_factor: int = 2
    max_training_steps: int | None = None

    use_cache: bool = True
    refresh_cache: bool = False

    compile_trunk: bool = False

    weak_edge_boost: float = 0.10
    refresh_batches: int = 1000

    principals_table: str = "principals"

    lr_schedule: str = "cosine"
    lr_warmup_steps: int = 0
    lr_warmup_ratio: float = 0.01
    lr_min_factor: float = 0.05

    hybrid_set_epochs: int = 1000
    hybrid_set_lr: float = 1e-3
    hybrid_set_weight_decay: float = 1e-4
    
    # Architecture
    hybrid_set_latent_dim: int = 128
    hybrid_set_hidden_dim: int = 1024
    hybrid_set_depth: int = 4
    hybrid_set_output_rank: int = 256
    
    # Loss Weights
    hybrid_set_w_bce: float = 1.0
    hybrid_set_w_count: float = 0.05
    hybrid_set_w_mass: float = 0.01    # New: Weight for "Energy" Mass Constraint

    # Focal Loss Gamma
    hybrid_set_focal_gamma: float = 4.0
    
    hybrid_set_save_interval: int = 400
    hybrid_set_recon_interval: int = 200
    hybrid_set_dropout: float = 0.0

    hybrid_set_heads: Dict[str, float] = field(default_factory=lambda: {
        "cast": 1.0,
        "director": 0.5,
        "writer": 0.5
    })

    joint_edge_tensor_cache: bool = True
    joint_edge_tensor_cache_file: str = "joint_edge_tensors.pt"


project_config = ProjectConfig()


def ensure_dirs(cfg: ProjectConfig):
    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.tensorboard_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)