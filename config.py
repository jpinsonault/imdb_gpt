from dataclasses import dataclass
from pathlib import Path

@dataclass
class ProjectConfig:
    data_dir: str = "data"
    db_path: str = "data/imdb.db"
    model_dir: str = "models"

    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    epochs: int = 10

    latent_dim: int = 512
    people_sequence_length: int = 8
    titles_sequence_length: int = 8

    save_interval: int = 500
    flush_interval: int = 250
    callback_interval: int = 200
    tensorboard_dir: str = "runs"
    log_interval: int = 50

    nce_temp: float = 0.07
    nce_weight: float = 1.0

    movie_limit: int = 100000000000

    num_workers: int = 0
    prefetch_factor: int = 0
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

    slot_people_count: int = 2
    slot_layers: int = 2
    slot_heads: int = 2
    slot_ff_mult: float = 2.0
    slot_dropout: float = 0.0
    slot_learning_rate: float = 1e-3
    slot_weight_decay: float = 1e-3
    slot_epochs: int = 10
    slot_log_interval: int = 50
    slot_save_interval: int = 1000
    slot_latent_align_weight: float = 0.1
    slot_diversity_weight: float = 0.01

    principals_table: str = "principals"

    slot_recon_interval: int = 20
    slot_recon_num_samples: int = 3
    slot_recon_show_slots: int = 3
    slot_recon_table_width: int = 60

project_config = ProjectConfig()

def ensure_dirs(cfg: ProjectConfig):
    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.tensorboard_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)
