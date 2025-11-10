from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectConfig:
    data_dir: str = "data"
    db_path: str = "data/imdb.db"
    model_dir: str = "models"

    batch_size: int = 2048  
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    epochs: int = 4

    latent_dim: int = 64

    save_interval: int = 500
    flush_interval: int = 250
    callback_interval: int = 200
    tensorboard_dir: str = "runs"
    log_interval: int = 50

    nce_temp: float = 0.07
    nce_weight: float = 1.0

    latent_type_loss_weight: float = 0.1

    movie_limit: int = 100000000000

    num_workers: int = 0
    prefetch_factor: int = 0
    max_training_steps: int | None = None

    use_cache: bool = True
    refresh_cache: bool = False

    compile_trunk: bool = False

    weak_edge_boost: float = 0.10
    refresh_batches: int = 1000

    principals_table: str = "principals"

    lr_schedule: str = "cosine"
    lr_warmup_steps: int = 0
    lr_warmup_ratio: float = 0.05
    lr_min_factor: float = 0.05

    path_siren_people_count: int = 10
    path_siren_lr: float = 0.001
    path_siren_weight_decay: float = 0
    path_siren_epochs: int = 100
    path_siren_layers: int = 10
    path_siren_hidden_mult: float = 4.0
    path_siren_omega0_first: float = 30.0
    path_siren_omega0_hidden: float = 1.0
    path_siren_callback_interval: int = 20
    path_siren_recon_num_samples: int = 2
    path_siren_table_width: int = 60
    path_siren_cache_capacity: int = 200000
    path_siren_save_interval: int = 500

    path_siren_movie_limit: int | None = None

    path_siren_loss_w_latent: float = 1.0
    path_siren_loss_w_title_latent: float = 1.0
    path_siren_loss_w_title_recon: float = 1.0
    path_siren_loss_w_latent_path: float = 1.0
    path_siren_loss_w_straight: float = 0.5
    path_siren_loss_w_curvature: float = 0.1

    path_siren_seed: int = 1337

    path_siren_time_fourier: int = 0

    image_ae_data_dir: str = "data/image_autoencoder"
    image_ae_runs_dir: str = "runs/image_autoencoder"
    image_ae_image_size: int = 128
    image_ae_in_channels: int = 3
    image_ae_base_channels: int = 32
    image_ae_latent_dim: int = 128
    image_ae_batch_size: int = 1
    image_ae_learning_rate: float = 1e-3
    image_ae_epochs: int = 100
    image_ae_recon_every: int = 2
    image_ae_max_recon_samples: int = 8


project_config = ProjectConfig()


def ensure_dirs(cfg: ProjectConfig):
    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.tensorboard_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.image_ae_data_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.image_ae_runs_dir).mkdir(parents=True, exist_ok=True)
