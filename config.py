# config.py

project_config = {
    'project_name': 'imdb_gpt',
    'data_dir': "./data/",
    'model_dir': 'models',
    'corpus_dir': "./data/corpus/",
    'docker_data_dir_mount': '/app/imdb',

    'autoencoder': {
        'latent_dim': 256,
        'batch_size': 512,
        'learning_rate': 0.001,
        'epochs': 40,
        'movies_file': 'movie.jsonl',
        'callback_interval': 100,
        'people_sequence_length': 1,
        'weight_decay': 1e-4,
        'early_stopping_patience': 10,
    }
}