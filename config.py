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
    'weight_decay': 1e-4,

    'people_sequence_length': 10,
    'latent_loss_weight': 1.0,
    'recon_loss_weight': 0.10,
    'latent_temperature': 0.03,

    # joint trainer specifics
    'nce_temp': 0.03,     # 
    'nce_weight': 1.0,    

    'edge_sampler': {
        'refresh_batches': 100,   
        'weak_edge_boost': 0.10,  
    },

    
    'tensorboard_dir': 'logs',
    'log_interval': 20,
    'callback_interval': 100,     
    'recon_log_interval': 20,     
    'row_recon_interval': 28,
    'row_recon_samples': 3,

    'movie_limit': 100_000_000_000,

    # i/o + runtime
    'db_path': './data/imdb.db',

    # checkpoints / flushing
    'save_interval': 10000,
    'flush_interval': 2000,

    # perf knobs
    'use_cuda_graphs': True,
    'compile_trunk': True,  # enable torch.compile for sequence predictor trunk
    'num_workers': 1,
    'prefetch_factor': 1,
}
