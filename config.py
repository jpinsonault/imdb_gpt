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
    lr_warmup_ratio: float = 0.05
    lr_min_factor: float = 0.05

    # Sequence Decoder Settings
    seq_decoder_epochs: int = 100
    seq_decoder_save_interval: int = 100
    seq_decoder_callback_interval: int = 100
    seq_decoder_recon_samples: int = 5
    seq_decoder_table_width: int = 120
    
    seq_decoder_len: int = 10 
    seq_decoder_hidden_dim: int = 256
    seq_decoder_layers: int = 6
    seq_decoder_heads: int = 8
    seq_decoder_dropout: float = 0.1
    seq_decoder_lr: float = 3e-4
    seq_decoder_weight_decay: float = 1e-2
    seq_decoder_w_latent: float = 1.0
    seq_decoder_w_recon: float = 1.0
    seq_decoder_w_presence: float = 1.0

    hybrid_set_epochs: int = 100
    hybrid_set_lr: float = 1e-3
    hybrid_set_weight_decay: float = 0.0 
    
    # Architecture
    hybrid_set_latent_dim: int = 128
    hybrid_set_hidden_dim: int = 1024
    hybrid_set_depth: int = 12
    hybrid_set_output_rank: int = 64
    hybrid_set_dropout: float = 0.0
    
    # Loss Weights
    hybrid_set_w_bce: float = 1.0
    hybrid_set_w_count: float = 0.05
    
    # [NEW] Focal Loss Gamma (Minimal Improvement)
    hybrid_set_focal_gamma: float = 4.0
    
    hybrid_set_save_interval: int = 500
    hybrid_set_recon_interval: int = 200

    hybrid_set_heads: Dict[str, float] = field(default_factory=lambda: {
        "cast": 1.0,
        "director": 0.5,
        "writer": 0.5
    })

    # Image AE / Siren (unchanged)
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
    image_ae_loss_w_l1: float = 1.0
    image_ae_loss_w_grad: float = 0.5
    image_ae_loss_w_tv: float = 0.0
    image_ae_loss_w_laplace: float = 0.5
    image_ae_latent_reg_weight: float = 0.0

    image_siren_runs_dir: str = "runs/image_siren"
    image_siren_hidden_dim: int = 256
    image_siren_hidden_layers: int = 20
    image_siren_w0_first: float = 30.0
    image_siren_w0_hidden: float = 1
    image_siren_lr: float = 1e-3
    image_siren_epochs: int = 500
    image_siren_batch_size: int = 6
    image_siren_recon_every: int = 10
    image_siren_max_recon_samples: int = 8
    image_siren_loss_w_l1: float = 1.0
    image_siren_loss_w_mse: float = 0.0
    image_siren_from_latent_sigmoid: bool = True
    image_siren_image_size: int = 256
    image_siren_eval_data_dir: str = "data/image_siren_eval"
    image_siren_samples_per_image: int = 4096
    image_siren_density_alpha: float = 0.9
    image_siren_uniform_frac: float = 0.1
    image_siren_image_repeats_per_epoch: int = 4

    joint_edge_tensor_cache: bool = True
    joint_edge_tensor_cache_file: str = "joint_edge_tensors.pt"


project_config = ProjectConfig()


def ensure_dirs(cfg: ProjectConfig):
    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.tensorboard_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.image_ae_data_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.image_ae_runs_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.image_siren_runs_dir).mkdir(parents=True, exist_ok=True)