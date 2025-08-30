# scripts/train_imdb_people_decoder.py
from config import project_config
from .autoencoder.imdb_row_autoencoders import TitlesAutoencoder, PeopleAutoencoder
from .autoencoder.one_to_many.trainer import build_sequence_logger, train_one_to_many
from .autoencoder.one_to_many.provider import ImdbMovieToPeopleProvider
from .autoencoder.training_callbacks import SequenceReconstructionLogger

def main():
    mov = TitlesAutoencoder(project_config)
    per = PeopleAutoencoder(project_config)
    mov.accumulate_stats(); mov.finalize_stats(); mov.build_autoencoder()
    per.accumulate_stats(); per.finalize_stats(); per.build_autoencoder()

    provider = ImdbMovieToPeopleProvider(
        db_path=project_config.db_path,
        seq_len=project_config.people_sequence_length,
    )

    seq_logger = build_sequence_logger(
        movie_ae=mov,
        people_ae=per,
        predictor=None,
        config=project_config,
        db_path=project_config.db_path,
        seq_len=project_config.people_sequence_length,
    )

    _ = train_one_to_many(
        config=project_config,
        provider=provider,
        steps=project_config.max_training_steps,
        save_every=0,
        seq_logger=seq_logger,
    )

if __name__ == "__main__":
    main()
