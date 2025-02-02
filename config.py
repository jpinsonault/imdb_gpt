# config.py


project_config = {
    'project_name': 'imdb_gpt',
    'data_dir': "./data/",
    'corpus_dir': "./data/corpus/",
    'docker_data_dir_mount': '/app/imdb',

    'dataset': {
        'jsonl_files': [
            "movie.jsonl",
            # "tvSeries.jsonl",
            # "person.jsonl",
        ],
    },
    
    'llm': {
        'input_length': 48,
        'character_embedding_dim': 256,
        'epochs': 100,
        'batches': 3000,
        'batch_size': 512,
        'num_heads': 8,
    },
    'autoencoder': {
        'latent_dim': 192,
        'batch_size': 512,
        'learning_rate': 0.00005,
        'epochs': 20,
        'movies_file': 'movie.jsonl',
        'callback_interval': 50,
    }
}