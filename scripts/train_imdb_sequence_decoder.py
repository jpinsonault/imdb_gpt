import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import sqlite3
import os
from contextlib import redirect_stdout
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    LayerNormalization,
    Conv1D,
    RepeatVector,
    Lambda,
    Activation,
    Add,
)
from tensorflow.keras import ops

from autoencoder.training_callbacks import ModelSaveCallback, TensorBoardPerBatchLoggingCallback, ReconstructionCallback, SequenceReconstructionCallback
from autoencoder.fields import (
    MaskedSparseCategoricalCrossentropy,
    TextField,
    ScalarField,
    MultiCategoryField,
    Scaling,
    BaseField
)
from scripts.autoencoder.imdb_row_autoencoders import PeopleAutoencoder, TitlesAutoencoder

@tf.keras.utils.register_keras_serializable()
def add_positional_encoding(input_sequence):
    seq_len = tf.shape(input_sequence)[1]
    model_dim = tf.shape(input_sequence)[2]

    positions = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)
    dims = tf.range(model_dim)
    angle_rates = 1 / tf.pow(10000.0, (2 * tf.cast(tf.math.floordiv(dims, 2), tf.float32)) / tf.cast(model_dim, tf.float32))
    angles = positions * angle_rates

    pos_encoding = tf.where(tf.math.floormod(dims, 2) == 0, tf.sin(angles), tf.cos(angles))
    pos_encoding = pos_encoding[tf.newaxis, ...]  # Expand batch dimension

    return input_sequence + pos_encoding


# -----------------------------------------------------------------------------
# Reusable 1×1 residual block
# -----------------------------------------------------------------------------

def residual_block(width: int, name: str | None = None):
    """Pre‑norm GELU residual block with kernel_size = 1.

    Args:
        width:     Number of channels for both conv layers and the residual add.
        name:      Optional base string for layer naming.
    Returns:
        A function mapping a tensor of shape (B, T, width) to the same shape.
    """
    def inner(x):
        res = x
        x = LayerNormalization(name=f"{name}_ln" if name else None)(x)
        x = Conv1D(width, 1, activation="gelu", name=f"{name}_conv1" if name else None)(x)
        x = Conv1D(width, 1, name=f"{name}_conv2" if name else None)(x)
        x = Add(name=f"{name}_add" if name else None)([x, res])
        x = Activation("gelu", name=f"{name}_act" if name else None)(x)
        return x

    return inner

class MoviesToPeopleSequenceAutoencoder:
    def __init__(self, config: Dict[str, Any], db_path: Path, model_dir: Path):
        self.config = config
        self.db_path = db_path
        self.model_dir = model_dir
        self.model = None
        self.stats_accumulated = False
        self.latent_dim = config["latent_dim"]
        self.people_sequence_length = config["people_sequence_length"]

        # instantiate and immediately load & freeze the pretrained row autoencoders
        self.movie_autoencoder_instance = TitlesAutoencoder(config, db_path, model_dir / "TitlesAutoencoder")
        self.movie_autoencoder_instance.load_model()
        self.movie_encoder = self.movie_autoencoder_instance.encoder
        self.movie_encoder.trainable = False

        self.people_autoencoder_instance = PeopleAutoencoder(config, db_path, model_dir / "PeopleAutoencoder")
        self.people_autoencoder_instance.load_model()
        self.people_decoder = self.people_autoencoder_instance.decoder
        self.people_decoder.trainable = False


    def accumulate_stats(self):
        # Accumulate stats using the instances.
        if not self.stats_accumulated:
            print("Accumulating stats for Movie fields...")
            self.movie_autoencoder_instance.accumulate_stats()
            print("Accumulating stats for People fields...")
            self.people_autoencoder_instance.accumulate_stats()
            self.stats_accumulated = True
            print("Stats accumulation complete for sequence model.")
        else:
            print("Stats already accumulated.")

    def row_generator(self):
        # (Keep existing generator - fetches movie and associated people)
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            conn.row_factory = sqlite3.Row
            movie_cursor = conn.cursor()
            # Use the same query as before (ensure title_genres join is correct)
            movie_cursor.execute("""
                SELECT
                    t.tconst, t.titleType, t.primaryTitle, t.startYear, t.endYear,
                    t.runtimeMinutes, t.averageRating, t.numVotes,
                    GROUP_CONCAT(g.genre, ',') AS genres
                FROM titles t
                INNER JOIN title_genres g ON t.tconst = g.tconst -- Use INNER JOIN
                WHERE t.startYear IS NOT NULL
                    AND t.averageRating IS NOT NULL AND t.runtimeMinutes IS NOT NULL
                    AND t.runtimeMinutes >= 5 AND t.startYear >= 1850
                    AND t.titleType IN ('movie', 'tvSeries', 'tvMovie', 'tvMiniSeries')
                    AND t.numVotes >= 10
                GROUP BY t.tconst
                -- LIMIT 100000 -- Consider removing limit
            """)

            people_query = """
                SELECT
                    p.primaryName, p.birthYear, p.deathYear,
                    GROUP_CONCAT(pp.profession, ',') AS professions
                FROM people p
                LEFT JOIN people_professions pp ON p.nconst = pp.nconst
                INNER JOIN principals pr ON pr.nconst = p.nconst
                WHERE pr.tconst = ? AND p.birthYear IS NOT NULL
                GROUP BY p.nconst
                HAVING COUNT(pp.profession) > 0
                ORDER BY pr.ordering -- Keep ordering
                LIMIT ?
            """

            for movie_row in movie_cursor:
                # Extract movie data relevant to movie fields
                movie_data = {
                    field.name: movie_row[field.name]
                    for field in self.movie_autoencoder_instance.fields
                    if field.name in movie_row.keys() # Ensure key exists
                }
                # Handle list field specifically
                if "genres" in movie_data and movie_data["genres"]:
                     movie_data["genres"] = movie_data["genres"].split(',')
                elif "genres" in movie_data:
                     movie_data["genres"] = [] # Handle case where genres might be NULL/empty string


                movie_tconst = movie_row["tconst"]
                people_cursor = conn.cursor()
                people_cursor.execute(people_query, (movie_tconst, self.people_sequence_length))

                people_list = []
                for people_row in people_cursor.fetchall():
                    # Extract people data relevant to people fields
                    person_data = {
                        field.name: people_row[field.name]
                        for field in self.people_autoencoder_instance.fields
                        if field.name in people_row.keys() # Ensure key exists
                    }
                    # Handle list field specifically
                    if "professions" in person_data and person_data["professions"]:
                         person_data["professions"] = person_data["professions"].split(',')
                    elif "professions" in person_data:
                         person_data["professions"] = None # Pass None if empty

                    people_list.append(person_data)

                # Yield combined data
                yield {**movie_data, "people": people_list}

    def build_autoencoder(self) -> tf.keras.Model:
        if not self.stats_accumulated:
            raise RuntimeError("Stats must be accumulated before building the model.")

        movie_fields  = self.movie_autoencoder_instance.fields
        people_fields = self.people_autoencoder_instance.fields
        latent_dim    = self.latent_dim
        seq_len       = self.people_sequence_length

        # movie inputs & frozen encoder
        movie_inputs = {
            f.name: tf.keras.Input(
                shape=f.input_shape,
                name=f"{f.name}_input",
                dtype=f.input_dtype,
            )
            for f in movie_fields
        }
        movie_latent = self.movie_encoder([movie_inputs[f.name] for f in movie_fields])

        trunk_width = latent_dim * 2
        x = RepeatVector(seq_len, name="repeat_latent")(movie_latent)  # (B, T, D)
        x = Conv1D(trunk_width, 1, activation="gelu", name="project_up")(x)

        for i in range(4):
            x = residual_block(trunk_width, name=f"resblock_{i}")(x)

        # final projection back to latent_dim
        x = Conv1D(latent_dim, 1, activation="gelu", name="project_down")(x)

        # ------------------------------------------------------------------
        # decode each timestep with the frozen people decoder
        # ------------------------------------------------------------------
        def _decode_and_reshape(z):
            b = tf.shape(z)[0]
            T = seq_len
            D = tf.shape(z)[2]
            flat = tf.reshape(z, (b * T, D))
            decoded = self.people_decoder(flat)

            def _reshape_tensor(t):
                tail = tf.shape(t)[1:]
                return tf.reshape(t, tf.concat([[b, T], tail], axis=0))

            return tf.nest.map_structure(_reshape_tensor, decoded)

        decoded_seq = Lambda(_decode_and_reshape, name="SequenceDecode")(x)

        # 4) collect and name outputs
        outputs = []
        for field in people_fields:
            rec = decoded_seq[f"{field.name}_decoder"]
            if isinstance(rec, (list, tuple)):
                main_seq, flag_seq = rec
                outputs.append(
                    Activation("linear", name=f"{field.name}_main_out")(main_seq)
                )
                outputs.append(
                    Activation("linear", name=f"{field.name}_flag_out")(flag_seq)
                )
            else:
                outputs.append(
                    Activation("linear", name=f"{field.name}_out")(rec)
                )

        # 5) build, assign, and return
        model = tf.keras.Model(
            inputs=list(movie_inputs.values()),
            outputs=outputs,
            name="MovieToPeopleSequencePredictor"
        )
        self.model = model
        return model


    
    def get_loss_dict(self) -> Dict[str, Any]:
        """
        Returns a dict mapping each sequence‐output layer name to its loss function,
        based on self.people_autoencoder_instance.fields and the naming convention
        in build_autoencoder().
        """
        loss_dict = {}
        for field in self.people_autoencoder_instance.fields:
            # your new model names each non-optional head as "<field>_out"
            # and each optional one as "<field>_main_out" + "<field>_flag_out"
            if field.optional:
                # main head uses the field’s base loss
                loss_dict[f"{field.name}_main_out"] = field._get_loss()
                # flag head always BinaryCrossentropy
                loss_dict[f"{field.name}_flag_out"] = tf.keras.losses.BinaryCrossentropy()
            else:
                loss_dict[f"{field.name}_out"] = field._get_loss()
        return loss_dict

    def get_loss_weights_dict(self) -> Dict[str, float]:
        """
        Returns a dict mapping each sequence‐output layer name to its weight,
        mirroring get_loss_dict but pulling `field.weight`.
        """
        weights = {}
        for field in self.people_autoencoder_instance.fields:
            if field.optional:
                weights[f"{field.name}_main_out"] = field.weight
                weights[f"{field.name}_flag_out"] = field.weight * 0.5
            else:
                weights[f"{field.name}_out"] = field.weight
        return weights

    def _print_model_architecture(self):
        """Prints summaries relevant to the sequence prediction model."""
        if not self.model:
            print("Sequence prediction model not built yet.")
            return

        print("\n--- Movie Encoder Summary (Used as Input Processor) ---")
        if self.movie_encoder:
            self.movie_encoder.summary()
        else:
            print("Movie encoder part not available.")

        # Don't print movie field decoders - they aren't used


        print("\n--- Main Sequence Prediction Model Summary ---")
        self.model.summary() # Summary of the Movie->People model
        print("\n--- Model Output Names ---")
        print(self.model.output_names)
        print("\n--- Loss Dictionary (Sequence Model) ---")
        print(self.get_loss_dict())
        print("\n--- Loss Weights Dictionary (Sequence Model) ---")
        print(self.get_loss_weights_dict())

    def _build_dataset(self) -> tf.data.Dataset:
        seq_len = self.people_sequence_length
        movie_fields = self.movie_autoencoder_instance.fields
        people_fields = self.people_autoencoder_instance.fields
        batch_size = self.config["batch_size"]

        def gen():
            for row in self.row_generator():
                # 1) grab & fix-length your people list
                ppl = row["people"]
                # if too short, repeat the last person; if empty, skip
                if len(ppl) < seq_len:
                    if not ppl:
                        continue
                    ppl = ppl + [ppl[-1]] * (seq_len - len(ppl))
                else:
                    ppl = ppl[:seq_len]

                # 2) movie inputs
                x_inputs = [ f.transform(row[f.name]) for f in movie_fields ]

                # 3) people-sequence targets
                y_targets = []
                for f in people_fields:
                    # for each field, collect one tensor per person, then stack
                    seq = [ f.transform_target(person.get(f.name)) for person in ppl ]
                    y_targets.append(tf.stack(seq))

                yield tuple(x_inputs), tuple(y_targets)

        # build your TensorSpecs just like before…
        input_specs = [
            tf.TensorSpec(shape=f.input_shape, dtype=f.input_dtype)
            for f in movie_fields
        ]
        output_specs = []
        for f in people_fields:
            shape = (seq_len,) + f.output_shape
            output_specs.append(tf.TensorSpec(shape=shape, dtype=f.output_dtype))

        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=(tuple(input_specs), tuple(output_specs))
        )
        return ds.batch(batch_size)


    def fit(self):
        """ Accumulates stats, builds, compiles, and fits the sequence prediction model. """
        if not self.stats_accumulated:
            self.accumulate_stats()
        if self.model is None:
            self.build_autoencoder() # Builds the Movie->People model

        self._print_model_architecture()

        initial_lr = self.config['learning_rate']

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=initial_lr, # Initial LR
            # weight_decay=self.config["weight_decay"],
        )

        # Compile the model
        print("Compiling sequence prediction model...")
        loss_dict = self.get_loss_dict()
        loss_weights_dict = self.get_loss_weights_dict()

        # Check consistency between loss keys and model output names
        model_output_keys = set(self.model.output_names)
        loss_keys = set(loss_dict.keys())
        if loss_keys != model_output_keys:
            print("WARNING: Loss dictionary keys do not perfectly match model output names!")
            print("Model Outputs:", sorted(list(model_output_keys)))
            print("Loss Dict Keys:", sorted(list(loss_keys)))
            print("Missing in Losses:", sorted(list(model_output_keys - loss_keys)))
            print("Extra in Losses:", sorted(list(loss_keys - model_output_keys)))
            # Raise error or proceed with caution
            # raise ValueError("Mismatch between model outputs and loss dictionary keys.")

        self.model.compile(
            optimizer=optimizer,
            loss=loss_dict,
            loss_weights=loss_weights_dict

            # Add metrics if needed
        )
        
        print("Model compiled.")

        # Prepare dataset
        print("Preparing dataset...")
        ds = self._build_dataset() # Gets (inputs_tuple, people_targets_dict, people_weights_dict)
        print("Dataset prepared.")

        # Callbacks
        # Use a specific log dir for the sequence model
        log_dir = os.path.join(self.model_dir, "logs", f"{self.__class__.__name__}_fit", str(int(tf.timestamp().numpy())))
        os.makedirs(log_dir, exist_ok=True)

        # Checkpoint callback for the sequence model
        checkpoint_path = os.path.join(self.model_dir, f"{self.__class__.__name__}_epoch_{{epoch:02d}}.keras")
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=False,
            monitor='loss',
            mode='min',
            save_best_only=False, # Save every epoch or best
            save_freq='epoch'
        )

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='loss', # Monitor training loss
            patience=self.config.get("early_stopping_patience", 10),
            restore_best_weights=True,
            verbose=1
        )

        sequence_recon_callback = SequenceReconstructionCallback(
            sequence_model_instance=self, 
            num_samples=5,
            interval_batches=self.config["callback_interval"],
        )
        log_root = Path(self.config["log_dir"])
        log_dir  = log_root / f"{self.__class__.__name__}_fit"
        log_dir.mkdir(parents=True, exist_ok=True)

        tensorboard_callback = TensorBoardPerBatchLoggingCallback(
            log_dir=log_dir,
            log_interval=20,
        )

        print("Starting sequence prediction model training...")
        self.model.fit(
            ds,
            epochs=self.config["epochs"],
            callbacks=[
                tensorboard_callback,
                model_checkpoint_callback,
                early_stopping_callback,
                sequence_recon_callback,
            ]
            # Add validation_data if available
        )
        print("\nTraining complete.")
        self.save_model() # Save the sequence model


##############################################################################
# Example Usage (Modified to use the sequence model)
##############################################################################

def main():
    # Assume project_config is imported
    from config import project_config
    # Use the 'autoencoder' sub-config for the sequence model too
    db_path = Path(project_config["data_dir"]) / "imdb.db"
    # Main directory for saving the sequence model and its components
    sequence_model_dir = Path(project_config["model_dir"]) / "SequencePredictor"

    print("\n--- Training Movie-to-People Sequence Predictor ---")
    sequence_predictor = MoviesToPeopleSequenceAutoencoder(project_config, db_path, sequence_model_dir)
    sequence_predictor.fit()


if __name__ == "__main__":
    # Basic logging setup
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
