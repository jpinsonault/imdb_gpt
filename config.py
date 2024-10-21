from pathlib import Path


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
        'input_length': 96,
        'character_embedding_dim': 128,
        'epochs': 4,
        'batches': 3000,
        'batch_size': 512,
        'num_heads': 8,
    },
}