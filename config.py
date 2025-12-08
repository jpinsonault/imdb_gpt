# config.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

@dataclass
class ProjectConfig:
    data_dir: str = "data"
    
    raw_db_path: str = "data/raw.db"
    db_path: str = "data/imdb.db"
    
    model_dir: str = "models"

    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 4
    log_interval: int = 50

    latent_dim: int = 64

    save_interval: int = 500
    tensorboard_dir: str = "runs"
    
    movie_limit: int = 100000000000
    principals_table: str = "principals"
    
    lr_schedule: str = "cosine"
    lr_warmup_steps: int = 1000
    lr_warmup_ratio: float = 0.0
    lr_min_factor: float = 0.05

    hybrid_set_epochs: int = 100
    hybrid_set_lr: float = 1e-3
    hybrid_set_weight_decay: float = 0.0
    
    hybrid_set_movie_dim: int = 256
    hybrid_set_hidden_dim: int = 512
    hybrid_set_person_dim: int = 256
    
    hybrid_set_w_bce: float = 1.0
    hybrid_set_w_recon: float = 0.5
    
    hybrid_set_logit_scale: float = 20.0

    hybrid_set_save_interval: int = 500
    hybrid_set_recon_interval: int = 200
    hybrid_set_dropout: float = 0.1

    hybrid_set_heads: Dict[str, float] = field(default_factory=lambda: {
        "cast": 1.0,
        "director": 0.5,
        "writer": 0.5
    })

    hybrid_set_head_groups: Dict[str, int] = field(default_factory=lambda: {
        "cast": 20,
        "director": 4,
        "writer": 8
    })

    use_cache: bool = True
    refresh_cache: bool = False


project_config = ProjectConfig()


def ensure_dirs(cfg: ProjectConfig):
    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.tensorboard_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)
