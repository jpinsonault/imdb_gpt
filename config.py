class ProjectConfig:
    """Simple project configuration."""
    def __init__(self):
        self.project_name = "imdb_gpt"
        self.data_dir = "./data/"
        self.log_dir = "logs"
        self.model_dir = "models"
        self.corpus_dir = "./data/corpus/"
        self.docker_data_dir_mount = "/app/imdb"

        self.latent_dim = 256
        self.batch_size = 512
        self.learning_rate = 0.0005
        self.weight_decay = 1e-4

        self.people_sequence_length = 10
        self.latent_loss_weight = 1.0
        self.recon_loss_weight = 0.10
        self.latent_temperature = 0.03

        self.nce_temp = 0.03
        self.nce_weight = 1.0

        self.refresh_batches = 100
        self.weak_edge_boost = 0.10

        self.tensorboard_dir = "logs"
        self.log_interval = 20
        self.callback_interval = 100
        self.recon_log_interval = 20
        self.row_recon_interval = 28
        self.row_recon_samples = 3

        self.movie_limit = 100000000000
        self.db_path = "./data/imdb.db"

        self.save_interval = 10000
        self.flush_interval = 2000

        self.use_cuda_graphs = True
        self.compile_trunk = True
        self.num_workers = 1
        self.prefetch_factor = 1

        self.use_cache = True
        self.refresh_cache = False
        self.epochs = 10
        self.input_length = 4096

    def to_dict(self):
        return dict(self.__dict__)


project_config = ProjectConfig()
