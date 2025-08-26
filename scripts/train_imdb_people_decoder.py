# scripts/train_one_to_many_examples.py
from config import project_config
from scripts.autoencoder.imdb_row_autoencoders import TitlesAutoencoder, PeopleAutoencoder
from scripts.autoencoder.one_to_many.trainer import train_one_to_many
from scripts.autoencoder.one_to_many.provider import ImdbMovieToPeopleProvider, ImdbPeopleToMovieProvider
from scripts.autoencoder.training_callbacks import SequenceReconstructionLogger

def main():
    mov = TitlesAutoencoder(project_config)
    per = PeopleAutoencoder(project_config)
    mov.accumulate_stats(); mov.finalize_stats(); mov.build_autoencoder()
    per.accumulate_stats(); per.finalize_stats(); per.build_autoencoder()

    p_movie_people = ImdbMovieToPeopleProvider(
        db_path=project_config.db_path,
        seq_len=project_config.people_sequence_length,
    )
    seq_logger = SequenceReconstructionLogger(
        movie_ae=mov,
        people_ae=per,
        predictor=None,
        db_path=project_config.db_path,
        seq_len=project_config.people_sequence_length,
        interval_steps=project_config.callback_interval,
        num_samples=2,
        table_width=38,
    )
    m2p = train_one_to_many(
        config=project_config,
        provider=p_movie_people,
        source_ae=mov,
        target_ae=per,
        steps=project_config.max_training_steps,
        save_every=0,
        seq_logger=seq_logger,
    )

if __name__ == "__main__":
    main()
