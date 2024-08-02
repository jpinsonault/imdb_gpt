from pathlib import Path


project_config = {
    'project_name'         : 'imdb_gpt',
    'data_dir'             : "./data/",
    'corpus_dir'           : "./data/corpus/",
    'docker_data_dir_mount': '/app/imdb',
    
    'entities': {
        'max_entity_length'  : 96,
        'title_min_rating'   : 1.0,
        'title_min_num_votes': 1000,
    },
    
    'search_autoencoder': {
        'epochs'                 : 16,
        'character_embedding_dim': 48,
        'batch_size'             : 128,
    },

    'llm': {
        'input_length'           : 128,
        'epochs'                 : 16,
        'character_embedding_dim': 256,
        'batch_size'             : 1024,
        'dataset_text_file'      : 'TinyStoriesV2-GPT4-train.txt',
    },
}