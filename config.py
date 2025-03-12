# config.py


project_config = {
    'project_name': 'imdb_gpt',
    'data_dir': "./data/",
    'model_dir': 'models',
    'corpus_dir': "./data/corpus/",
    'docker_data_dir_mount': '/app/imdb',

    'autoencoder': {
        'latent_dim': 192,
        'batch_size': 512,
        'learning_rate': 0.00005,
        'epochs': 40,
        'movies_file': 'movie.jsonl',
        'callback_interval': 200,
    }
}