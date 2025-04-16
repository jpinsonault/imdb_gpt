import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import sqlite3
import os
from contextlib import redirect_stdout
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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
from scripts.autoencoder.row_autoencoder import RowAutoencoder

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

        movie_fields = self.movie_autoencoder_instance.fields
        people_fields = self.people_autoencoder_instance.fields
        latent_dim = self.latent_dim
        seq_len = self.people_sequence_length

        # — movie inputs & frozen encoder —
        movie_inputs = {
            field.name: tf.keras.Input(shape=field.input_shape, name=f"{field.name}_input", dtype=field.input_dtype)
            for field in movie_fields
        }
        # pass them *in the same order* the TitlesAutoencoder.encoder expects
        movie_latent = self.movie_encoder([
            movie_inputs[field.name] for field in movie_fields
        ])

        # — expand to sequence, process, reshape for decoding —
        x = tf.keras.layers.RepeatVector(seq_len)(movie_latent)
        x = tf.keras.layers.Conv1D(latent_dim * 2, 1, activation='linear')(x)
        x = tf.keras.layers.Conv1D(latent_dim,   1, activation='gelu')(x)
        x = tf.keras.layers.LayerNormalization()(x)
        flat = tf.reshape(x, (-1, latent_dim))   # (batch*seq_len, latent_dim)

        # — run the frozen people‐decoder on every time step in batch*
        decoder_outputs = self.people_decoder(flat)
        # decoder_outputs is a dict: e.g. { "primaryName_decoder": tensor or [main,flag], ... }

        outputs = []
        names   = []
        T = seq_len
        for field in people_fields:
            recon = decoder_outputs[f"{field.name}_decoder"]
            if isinstance(recon, (list, tuple)):
                main_flat, flag_flat = recon
            else:
                main_flat, flag_flat = recon, None

            # reshape back into (batch, T, …)
            main_seq = tf.reshape(main_flat, (-1, T) + tuple(main_flat.shape[1:]))
            name = f"{field.name}_main_out"
            outputs.append(tf.keras.layers.Activation('linear', name=name, dtype=tf.float32)(main_seq))
            names.append(name)

            if flag_flat is not None:
                flag_seq = tf.reshape(flag_flat, (-1, T, 1))
                fname = f"{field.name}_flag_out"
                outputs.append(tf.keras.layers.Activation('linear', name=fname, dtype=tf.float32)(flag_seq))
                names.append(fname)

        self.model = tf.keras.Model(
            inputs=list(movie_inputs.values()),
            outputs=outputs,
            name="MovieToPeopleSequencePredictor"
        )
        return self.model


    def _build_dataset(self) -> tf.data.Dataset:
        """
        Builds the tf.data.Dataset for training the sequence prediction model.

        Yields tuples of (inputs, targets, sample_weights):
        - inputs: A tuple of tensors representing the input movie fields.
        - targets: A dictionary mapping PEOPLE output layer names to the corresponding
                   target tensors (people field sequences). Handles main/flag.
        - sample_weights: A dictionary mapping PEOPLE output layer names to sample weights
                          (masking padded time steps).

        Raises:
            RuntimeError: If stats have not been accumulated before calling.
        """
        if not self.stats_accumulated:
            raise RuntimeError("Stats must be accumulated before building the dataset. Call accumulate_stats().")

        # Get necessary attributes
        movie_fields: List[BaseField] = self.movie_autoencoder_instance.fields
        people_fields: List[BaseField] = self.people_autoencoder_instance.fields
        seq_length: int = self.people_sequence_length
        batch_size = self.config.get("batch_size", 32)

        # Ensure people decoders were built
        if not hasattr(self, 'people_decoders') or not self.people_decoders:
             raise RuntimeError("People decoder instances not found. Ensure build_autoencoder() has been called.")
        people_decoders = self.people_decoders

        def db_generator():
            """Generator function yielding data for sequence prediction."""
            for row_dict in self.row_generator():
                # --- 1. Process Inputs (Movie Fields) ---
                inputs_tuple = []
                try:
                     inputs_tuple = tuple(field.transform(row_dict.get(field.name)) for field in movie_fields)
                except Exception as e:
                    logging.error(f"Error transforming INPUT fields: {e}. Row: {row_dict}", exc_info=True)
                    continue # Skip problematic row

                # --- 2. Process ONLY People Targets and Sample Weights ---
                people_targets = {}
                people_sample_weights = {}
                people_seq_list = row_dict.get("people", [])
                num_real_people = len(people_seq_list)

                for field_idx, field in enumerate(people_fields): # Use index for potentially better error reporting
                    try:
                        main_target_list = []
                        flag_target_list = []
                        decoder = people_decoders[field.name]
                        decoder_is_multi_output = isinstance(decoder.output, list)
                        transform_target_returns_tuple = None

                        for i in range(seq_length):
                            is_padding_step = (i >= num_real_people)
                            main_target_part = None
                            flag_target_part = None # Reset for each step

                            if is_padding_step:
                                # --- Handle PADDING steps ---
                                # *** MODIFICATION START ***
                                if field.optional:
                                    # Optional fields handle None via transform_target
                                    target_result = field.transform_target(None) # Returns (padded_base, flag=1.0)
                                    if not isinstance(target_result, tuple) or len(target_result) != 2:
                                        raise ValueError(f"Optional field {field.name}.transform_target(None) did not return tuple of 2")
                                    main_target_part, flag_target_part = target_result
                                    if transform_target_returns_tuple is None: transform_target_returns_tuple = True
                                else:
                                    # Non-optional fields: Directly get base padding value. No flag involved.
                                    main_target_part = field.get_base_padding_value()
                                    flag_target_part = None # Explicitly None
                                    if transform_target_returns_tuple is None: transform_target_returns_tuple = False
                                # *** MODIFICATION END ***
                            else:
                                # --- Handle REAL data steps ---
                                raw_val = people_seq_list[i].get(field.name)
                                target_result = field.transform_target(raw_val) # Returns (transformed, flag=0.0) or just transformed

                                # Unpack based on field optionality
                                if field.optional:
                                    if not isinstance(target_result, tuple) or len(target_result) != 2:
                                         raise ValueError(f"Optional field {field.name}.transform_target() did not return tuple of 2 for value: {raw_val}")
                                    main_target_part, flag_target_part = target_result
                                    if transform_target_returns_tuple is None: transform_target_returns_tuple = True
                                else:
                                    main_target_part = target_result
                                    flag_target_part = None
                                    if transform_target_returns_tuple is None: transform_target_returns_tuple = False


                            # Consistency check after first step
                            if transform_target_returns_tuple is None:
                                 raise RuntimeError(f"Logic error: transform_target_returns_tuple not set for field {field.name}")


                            # Append results to lists
                            main_target_list.append(main_target_part)

                            # Append flag target *only if the decoder expects it*
                            if decoder_is_multi_output:
                                if transform_target_returns_tuple: # Flag was provided by transform_target (i.e., field is optional)
                                    flag_target_list.append(tf.cast(flag_target_part, tf.float32))
                                else: # Decoder needs flag, but field isn't optional
                                    # Generate default flag (0.0 for real, 1.0 for padding)
                                    default_flag = tf.cast(1.0 if is_padding_step else 0.0, tf.float32)
                                    flag_target_list.append(default_flag)


                        # Stack the lists into sequence tensors
                        stacked_main_seq = tf.stack(main_target_list)

                        # Create weights for time steps (masking padding)
                        time_step_weights = tf.concat([
                            tf.ones(num_real_people, dtype=tf.float32),
                            tf.zeros(seq_length - num_real_people, dtype=tf.float32)
                        ], axis=0)

                        # Assign to targets and sample_weights dicts using correct names
                        if decoder_is_multi_output:
                            if len(flag_target_list) != seq_length:
                                raise ValueError(f"Flag target list length mismatch for {field.name}")
                            stacked_flag_seq = tf.stack(flag_target_list)
                            main_name = f"{field.name}_main_out"
                            flag_name = f"{field.name}_flag_out"
                            people_targets[main_name] = stacked_main_seq
                            people_targets[flag_name] = stacked_flag_seq
                            people_sample_weights[main_name] = time_step_weights
                            people_sample_weights[flag_name] = tf.ones_like(time_step_weights)
                        else: # Single output decoder
                            out_name = f"{field.name}_out"
                            people_targets[out_name] = stacked_main_seq
                            people_sample_weights[out_name] = time_step_weights # Apply weights unless loss handles masking


                    except Exception as e:
                        logging.error(f"Error processing TARGET for people field #{field_idx} ('{field.name}'): {e}. Row: {row_dict}", exc_info=True)
                        # Decide whether to skip row or raise
                        raise e # Re-raise to stop processing

                # --- 3. Yield the structured data ---
                yield inputs_tuple, people_targets, people_sample_weights


        # --- Define Output Signature --- (Copied from previous correct version)
        input_specs = tuple(tf.TensorSpec(shape=field.input_shape, dtype=field.input_dtype)
                            for field in movie_fields)

        target_specs = {}
        for field in people_fields:
            decoder = people_decoders[field.name]
            is_multi_output = isinstance(decoder.output, list)
            base_shape = field._get_output_shape()
            base_dtype = field.output_dtype
            flag_shape = (1,)
            flag_dtype = tf.float32
            seq_base_shape = (seq_length,) + base_shape
            seq_flag_shape = (seq_length,) + flag_shape

            if is_multi_output:
                target_specs[f"{field.name}_main_out"] = tf.TensorSpec(shape=seq_base_shape, dtype=base_dtype)
                target_specs[f"{field.name}_flag_out"] = tf.TensorSpec(shape=seq_flag_shape, dtype=flag_dtype)
            else:
                target_specs[f"{field.name}_out"] = tf.TensorSpec(shape=seq_base_shape, dtype=base_dtype)

        sample_weight_specs = {}
        for field in people_fields:
            decoder = people_decoders[field.name]
            is_multi_output = isinstance(decoder.output, list)
            seq_weight_spec = tf.TensorSpec(shape=(seq_length,), dtype=tf.float32)
            if is_multi_output:
                sample_weight_specs[f"{field.name}_main_out"] = seq_weight_spec
                sample_weight_specs[f"{field.name}_flag_out"] = seq_weight_spec
            else:
                sample_weight_specs[f"{field.name}_out"] = seq_weight_spec

        # --- Create tf.data.Dataset ---
        ds = tf.data.Dataset.from_generator(
            db_generator,
            output_signature=(input_specs, target_specs, sample_weight_specs)
        )

        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds

    def get_loss_dict(self) -> Dict[str, Any]:
        """ Returns the loss dictionary ONLY for PEOPLE outputs. """
        loss_dict = {}
        if not hasattr(self, 'people_decoders') or not self.people_decoders:
             raise RuntimeError("People decoder instances not found for get_loss_dict. Ensure build_autoencoder() has been called.")
        people_decoders = self.people_decoders

        # People losses ONLY
        for field in self.people_autoencoder_instance.fields:
            decoder = people_decoders.get(field.name)
            if decoder is None:
                raise ValueError(f"Decoder for people field '{field.name}' not found.")
            is_multi_output = isinstance(decoder.output, list)
            base_loss = field._get_loss() # Get the base loss instance (MSE, BCE, MaskedSCCE)

            if is_multi_output:
                # Assign appropriate losses to main and flag outputs
                main_name = f"{field.name}_main_out"
                flag_name = f"{field.name}_flag_out"
                loss_dict[main_name] = base_loss
                # Flag loss is typically BCE
                loss_dict[flag_name] = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
            else:
                out_name = f"{field.name}_out"
                loss_dict[out_name] = base_loss

        # Ensure all model outputs (which are now only people outputs) have a loss
        if self.model:
            for output_name in self.model.output_names:
                if output_name not in loss_dict:
                    # This could happen if model output names differ from expected keys
                    logging.warning(f"Model Output '{output_name}' is missing from the generated loss dictionary. Keys: {list(loss_dict.keys())}")
                    # Attempt to find a match based on field name? Or raise error.
                    # raise ValueError(f"Output '{output_name}' is missing from the loss dictionary.")

        return loss_dict

    def get_loss_weights_dict(self) -> Dict[str, float]:
        """ Returns the loss weights dictionary ONLY for PEOPLE outputs. """
        weights_dict = {}
        if not hasattr(self, 'people_decoders') or not self.people_decoders:
             raise RuntimeError("People decoder instances not found for get_loss_weights_dict.")
        people_decoders = self.people_decoders

        # People weights ONLY
        for field in self.people_autoencoder_instance.fields:
            decoder = people_decoders.get(field.name)
            if decoder is None:
                 raise ValueError(f"Decoder for people field '{field.name}' not found.")
            is_multi_output = isinstance(decoder.output, list)
            weight = field.weight # Get the original field weight

            if is_multi_output:
                main_name = f"{field.name}_main_out"
                flag_name = f"{field.name}_flag_out"
                # Assign weights - apply base weight to main, maybe reduced to flag
                weights_dict[main_name] = weight
                weights_dict[flag_name] = weight * 0.5 # Example: Reduce flag weight
            else:
                out_name = f"{field.name}_out"
                weights_dict[out_name] = weight

        return weights_dict

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

        print("\n--- People Field Decoder Summaries (Used in Sequence Generation) ---")
        if self.people_decoders:
             for name, decoder in self.people_decoders.items():
                 print(f"\nPeople Decoder for field: {name}")
                 decoder.summary()
        else:
             print("People decoders not available.")

        print("\n--- Main Sequence Prediction Model Summary ---")
        self.model.summary() # Summary of the Movie->People model
        print("\n--- Model Output Names ---")
        print(self.model.output_names)
        print("\n--- Loss Dictionary (Sequence Model) ---")
        print(self.get_loss_dict())
        print("\n--- Loss Weights Dictionary (Sequence Model) ---")
        print(self.get_loss_weights_dict())


    def fit(self):
        """ Accumulates stats, builds, compiles, and fits the sequence prediction model. """
        if not self.stats_accumulated:
            self.accumulate_stats()
        if self.model is None:
            self.build_autoencoder() # Builds the Movie->People model

        self._print_model_architecture()

        initial_lr = self.config['learning_rate']
        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )

        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=initial_lr, # Initial LR
            weight_decay=self.config.get("weight_decay", 1e-4)
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
            loss=loss_dict, # Pass the dictionary
            loss_weights=loss_weights_dict # Pass the dictionary
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
            db_path=self.db_path,
            num_samples=3,
            interval_batches=self.config["callback_interval"],
        )
        tensorboard_callback = TensorBoardPerBatchLoggingCallback(log_dir=log_dir, log_interval=20)

        print("Starting sequence prediction model training...")
        self.model.fit(
            ds,
            epochs=self.config["epochs"],
            callbacks=[
                tensorboard_callback,
                lr_callback,
                model_checkpoint_callback,
                early_stopping_callback,
                sequence_recon_callback,
            ]
            # Add validation_data if available
        )
        print("\nTraining complete.")
        self.save_model() # Save the sequence model

    def save_model(self):
        """ Saves the main sequence prediction model and the movie encoder part. """
        if not self.model:
            print("Sequence prediction model not built. Cannot save.")
            return

        # Ensure model_dir exists
        output_dir = Path(self.model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the main sequence prediction model
        seq_model_path = output_dir / f"{self.__class__.__name__}_predictor_final.keras"
        print(f"Saving final sequence predictor model to {seq_model_path}")
        self.model.save(seq_model_path)

        # Save the movie encoder part separately (optional but potentially useful)
        if self.movie_encoder:
            enc_path = output_dir / f"{self.__class__.__name__}_movie_encoder_final.keras"
            print(f"Saving final movie encoder model (from sequence model) to {enc_path}")
            self.movie_encoder.save(enc_path)
        else:
             print("Movie encoder instance not found, skipping saving.")

        print("Sequence models saved.")

    def load_model(self, compile_model=False):
        """ Loads the saved sequence prediction model and its movie encoder. """
        # Required for custom objects
        # tf.keras.config.enable_unsafe_deserialization() # Use if needed, Keras 3 might handle better

        # Define custom objects needed for loading
        # Make sure ALL custom classes/functions used in the model are included
        custom_objects = {
            "MaskedSparseCategoricalCrossentropy": MaskedSparseCategoricalCrossentropy,
            "add_positional_encoding": add_positional_encoding,
        }

        input_dir = Path(self.model_dir)
        seq_model_path = input_dir / f"{self.__class__.__name__}_predictor_final.keras"
        enc_path = input_dir / f"{self.__class__.__name__}_movie_encoder_final.keras"

        if not seq_model_path.exists():
            raise FileNotFoundError(f"Sequence predictor model file not found: {seq_model_path}")

        print(f"Loading sequence predictor model from {seq_model_path}...")
        self.model = tf.keras.models.load_model(
            seq_model_path,
            custom_objects=custom_objects,
            compile=compile_model # Compile only if continuing training
        )
        print("Sequence predictor model loaded.")

        if enc_path.exists():
             print(f"Loading movie encoder model from {enc_path}...")
             self.movie_encoder = tf.keras.models.load_model(
                 enc_path,
                 custom_objects=custom_objects,
                 compile=False # Encoder typically doesn't need compiling standalone
             )
             print("Movie encoder model loaded.")
        else:
             print(f"Movie encoder file not found at {enc_path}, skipping load.")
             self.movie_encoder = None # Ensure it's None if not loaded



##############################################################################
# Example Usage (Modified to use the sequence model)
##############################################################################

def main():
    # Assume project_config is imported
    from config import project_config
    # Use the 'autoencoder' sub-config for the sequence model too
    config = project_config["autoencoder"]
    db_path = Path(project_config["data_dir"]) / "imdb.db"
    # Main directory for saving the sequence model and its components
    sequence_model_dir = Path(project_config["model_dir"]) / "SequencePredictor"

    print("\n--- Training Movie-to-People Sequence Predictor ---")
    sequence_predictor = MoviesToPeopleSequenceAutoencoder(config, db_path, sequence_model_dir)
    sequence_predictor.fit()


if __name__ == "__main__":
    # Basic logging setup
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
