# config.py
from dataclasses import dataclass, asdict

@dataclass
class ProjectConfig:
    project_name: str = "imdb_gpt"
    data_dir: str = "./data/"
    log_dir: str = "logs"
    model_dir: str = "models"
    corpus_dir: str = "./data/corpus/"
    docker_data_dir_mount: str = "/app/imdb"

    latent_dim: int = 256
    batch_size: int = 512
    learning_rate: float = 0.0005
    weight_decay: float = 1e-4

    people_sequence_length: int = 10
    latent_loss_weight: float = 1.0
    recon_loss_weight: float = 0.10
    latent_temperature: float = 0.03

    nce_temp: float = 0.03
    nce_weight: float = 1.0

    refresh_batches: int = 100
    weak_edge_boost: float = 0.10

    tensorboard_dir: str = "logs"
    log_interval: int = 20
    callback_interval: int = 100
    recon_log_interval: int = 20
    row_recon_interval: int = 28
    row_recon_samples: int = 3

    movie_limit: int = 100000000000
    db_path: str = "./data/imdb.db"

    save_interval: int = 10000
    flush_interval: int = 2000

    use_cuda_graphs: bool = True
    compile_trunk: bool = True
    num_workers: int = 1
    prefetch_factor: int = 1

    use_cache: bool = True
    refresh_cache: bool = False
    epochs: int = 10
    input_length: int = 4096

    def to_dict(self):
        return asdict(self)

project_config = ProjectConfig()
