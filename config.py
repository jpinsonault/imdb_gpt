# config.py

project_config = {
    'project_name': 'imdb_gpt',
    'data_dir': "./data/",
    'log_dir': 'logs',
    'model_dir': 'models',
    'corpus_dir': "./data/corpus/",
    'docker_data_dir_mount': '/app/imdb',

    'latent_dim': 256,
    'batch_size': 512,
    'learning_rate': 0.0005,
    'epochs': 20,
    'callback_interval': 100,
    'people_sequence_length': 1,
    'weight_decay': 1e-4,
    'early_stopping_patience': 10,
    'movie_limit': 1000000000,
    'db_path': './data/imdb.db',
    'nce_temp': 0.03,
    'weak_sample_boost': 0.01,
    'edge_sampler': {
        'refresh_batches': 100,
        'weak_edge_boost': 0.10,
    },
}