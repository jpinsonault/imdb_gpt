project_config = {
    'project_name': 'imdb_gpt',
    'data_dir': "./data/",
    'log_dir': 'logs',
    'model_dir': 'models',
    'corpus_dir': "./data/corpus/",
    'docker_data_dir_mount': '/app/imdb',

    'latent_dim': 256,
    'batch_size': 1024,
    'learning_rate': 0.0005,
    'weight_decay': 1e-4,

    'reconstruction_interval': 100,
    'callback_interval': 100,
    'people_sequence_length': 1,
    'movie_limit': 1000000000,
    'db_path': './data/imdb.db',
    'nce_temp': 0.03,

    'edge_sampler': {
        'refresh_batches': 100,
        'weak_edge_boost': 0.10,
    },

    'tensorboard_dir': 'logs',

    'log_interval': 20,
    'recon_log_interval': 20,
    'row_recon_interval': 28,
    'row_recon_samples': 3,
    'save_interval': 10000,
    'flush_interval': 2000,

    'num_workers': 6,
    'prefetch_factor': 6,
}
