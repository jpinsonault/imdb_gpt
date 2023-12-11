from pathlib import Path


project_config = {
    'project_name': 'imdb_gpt',
    'data_dir'    : "g:/imdb",
    
    'entities': {
        'max_entity_length'  : 94,
        'title_min_rating'   : 1.0,
        'title_min_num_votes': 1000,
    },
    
    'search_autoencoder': {
        'batch_size'             : 2048,
        'epochs'                 : 2,
        'character_embedding_dim': 64,
        'latent_dim'             : 128,
        'batch_size'             : 2048,
    },
}