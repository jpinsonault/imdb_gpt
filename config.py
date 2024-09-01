from pathlib import Path


project_config = {
    'project_name'         : 'imdb_gpt',
    'data_dir'             : "./data/",
    'corpus_dir'           : "./data/corpus/",
    'docker_data_dir_mount': '/app/imdb',

    'dataset': {
        'jsonl_files': [
            "movie.jsonl",
            "movie.jsonl",
            "movie.jsonl",
            "movie.jsonl",
            "movie.jsonl",
            "movie.jsonl",
            "movie.jsonl",
            "movie.jsonl",
            "movie.jsonl",
            "movie.jsonl",
            # "tvSeries.jsonl",
            # "episode.jsonl",
            "person.jsonl",
            "character.jsonl",
        ],
    },
    
    'entities': {
        'max_entity_length'  : 96,
        'title_min_rating'   : 1.0,
        'title_min_num_votes': 1000,
    },
    
    'search_autoencoder': {
        'epochs'                  : 16,
        'character_embedding_dim' : 96,
        'batch_size'              : 64,
    },

    'siren': {
        'max_output_length'       : 288,
        'batch_size'              : 128,
        'character_embedding_dim' : 256,
        'bottleneck_dim'          : 256,
        'siren_width'             : 512,
        'num_encoder_blocks'      : 4,
        'num_siren_layers'        : 4,
    },

    'llm': {
        'input_length'            : 128,
        'epochs'                  : 16,
        'character_embedding_dim' : 192,
        'batch_size'              : 512,
        'dataset_text_file'       : 'TinyStoriesV2-GPT4-train.txt',
    },
}