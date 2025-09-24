from dataclasses import dataclass
from pathlib import Path

@dataclass
class ProjectConfig:
    data_dir: str = "data"
    db_path: str = "data/imdb.db"
    model_dir: str = "models"

    batch_size: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    epochs: int = 1

    latent_dim: int = 512
    people_sequence_length: int = 16
    titles_sequence_length: int = 16

    nce_temp: float = 0.07
    nce_weight: float = 1.0

    save_interval: int = 500
    flush_interval: int = 250
    callback_interval: int = 200
    tensorboard_dir: str = "runs"
    log_interval: int = 50

    num_workers: int = 0
    prefetch_factor: int = 2
    max_training_steps: int | None = None

    use_cache: bool = True
    refresh_cache: bool = False

    compile_trunk: bool = False

    weak_edge_boost: float = 0.10
    refresh_batches: int = 1000

    alternate_modes: bool = True
    min_mode_frac: float = 0.25
    max_mode_frac: float = 0.75
    many_to_many_warm_start: bool = False
    many_to_many_freeze_loaded: bool = False

project_config = ProjectConfig()

def ensure_dirs(cfg: ProjectConfig):
    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.tensorboard_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)
