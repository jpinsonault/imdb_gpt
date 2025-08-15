project_config = {
    'project_name': 'imdb_gpt',
    'data_dir': 'data',
    'log_dir': 'logs',
    'model_dir': 'models',
    'corpus_dir': 'data/corpus',
    'docker_data_dir_mount': '/app/imdb',

    'db_path': 'data/imdb.db',

    'latent_dim': 256,
    'batch_size': 5000,
    'learning_rate': 0.0005,
    'weight_decay': 1e-4,

    'reconstruction_interval': 100,
    'callback_interval': 100,
    'people_sequence_length': 1,
    'movie_limit': 1000000000,

    'nce_temp': 0.03,
    'nce_weight': 1.0,

    'latent_loss_weight': 1.0,
    'contrastive_loss_weight': 0.5,

    'edge_sampler': {
        'weak_edge_boost': 0.10
    },

    'tensorboard_dir': 'logs',

    'log_interval': 20,
    'recon_log_interval': 100,
    'row_recon_interval': 100,
    'row_recon_samples': 3,
    'save_interval': 10000,
    'flush_interval': 2000,

    'num_workers': 8,
    'prefetch_factor': 1,

    'dataset_backend': 'memmap',
    'memstore_dir': 'data/memstore',
    'memstore_policy': 'rebuild',
    'memstore_chunk': 10000
}
