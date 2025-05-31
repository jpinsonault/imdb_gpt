import logging
import pickle
import sqlite3
import numpy as np
from pathlib import Path
from prettytable import PrettyTable
from typing import Any, List, Dict
import os
from tqdm import tqdm
import tensorflow as tf
from autoencoder.fields import BaseField, BooleanField, MaskedSparseCategoricalCrossentropy, MultiCategoryField, NumericDigitCategoryField, ScalarField, SingleCategoryField, add_positional_encoding, TextField
from autoencoder.training_callbacks import ModelSaveCallback, ReconstructionCallback, TensorBoardPerBatchLoggingCallback

# Alias for distributions
logging.basicConfig(level=logging.info, format='%(asctime)s - %(levelname)s - %(message)s')


def _attention_fuse(
        latents,
        latent_dim,
        num_heads=4,
        key_dim=None,
):
    if key_dim is None:
        key_dim = max(8, latent_dim // num_heads)

    proj_tokens = []
    for i, t in enumerate(latents):
        if t.shape[-1] != latent_dim:
            t = tf.keras.layers.Dense(latent_dim, name=f"field_proj_{i}")(t)
        t = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, 1))(t)
        proj_tokens.append(t)

    tokens = tf.keras.layers.Concatenate(axis=1, name="token_concat")(proj_tokens)

    attn_out = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        value_dim=latent_dim,
        output_shape=latent_dim,
        name="field_mha",
    )(tokens, tokens)

    x = tf.keras.layers.Add()([tokens, attn_out])
    x = tf.keras.layers.LayerNormalization()(x)

    pooled = tf.keras.layers.Lambda(lambda t: tf.reduce_mean(t, axis=1))(x)
    return tf.keras.layers.LayerNormalization(name="latent")(pooled)


@tf.keras.utils.register_keras_serializable("Custom", "BatchRepel")
class BatchRepel(tf.keras.layers.Layer):
    def __init__(self, coeff: float = 1e-2, **kw):
        super().__init__(**kw)
        self.coeff = coeff

    def call(self, z):
        z = tf.nn.l2_normalize(z, axis=-1)
        sim = tf.matmul(z, z, transpose_b=True)
        b = tf.shape(sim)[0]
        mask = tf.ones_like(sim) - tf.eye(b)
        off = sim * mask
        reg = tf.reduce_sum(tf.square(off)) / tf.cast(b * (b - 1), tf.float32)
        self.add_loss(self.coeff * reg)
        return z

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"coeff": self.coeff})
        return cfg


class RowAutoencoder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_dir = Path(config['model_dir'])
        self.model = None
        self.encoder = None
        self.decoder = None

        self.fields: List[BaseField] = self.build_fields()
        self.latent_dim = self.config["latent_dim"]
        self.num_rows_in_dataset = 0

        self.db_path: str = config["db_path"]
        self.stats_accumulated = False

    # --------------------------------------------------------------------- #
    # cache helpers
    # --------------------------------------------------------------------- #
    def _cache_table_name(self) -> str:
        return f"{self.__class__.__name__}_stats_cache"

    def _drop_cache_table(self):
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            conn.execute(f"DROP TABLE IF EXISTS {self._cache_table_name()};")
            conn.commit()

    def _save_cache(self):
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self._cache_table_name()} ("
                "field_name TEXT PRIMARY KEY, data BLOB)"
            )
            for f in self.fields:
                blob = pickle.dumps(f)
                conn.execute(
                    f"INSERT OR REPLACE INTO {self._cache_table_name()} "
                    "(field_name, data) VALUES (?, ?);",
                    (f.name, blob),
                )
            conn.commit()

    def _load_cache(self) -> bool:
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name=?;", (self._cache_table_name(),)
            )
            if cur.fetchone() is None:
                return False

            cur.execute(f"SELECT field_name, data FROM {self._cache_table_name()};")
            rows = cur.fetchall()
            if not rows:
                return False

            cache_map = {name: pickle.loads(blob) for name, blob in rows}
            for i, f in enumerate(self.fields):
                if f.name in cache_map:
                    self.fields[i] = cache_map[f.name]

        self.stats_accumulated = True
        return True

    # --------------------------------------------------------------------- #
    # public API
    # --------------------------------------------------------------------- #
    def accumulate_stats(
        self,
        use_cache: bool = True,
        refresh_cache: bool = False,
    ):
        if refresh_cache:
            self._drop_cache_table()

        if use_cache and self._load_cache():
            logging.info("stats loaded from cache")
            return

        if self.stats_accumulated:
            logging.info("stats already accumulated")
            return

        n = 0
        logging.info("accumulating stats")
        for row in tqdm(self.row_generator(), desc=self.__class__.__name__):
            self.accumulate_stats_for_row(row)
            n += 1
        self.num_rows_in_dataset = n
        logging.info(f"stats accumulation finished ({n} rows)")

    def finalize_stats(self):
        if self.stats_accumulated:
            return

        logging.info("finalizing stats")
        for f in self.fields:
            f.finalize_stats()
        self.stats_accumulated = True
        self._save_cache()
        logging.info("stats finalized and cached")

    def accumulate_stats_for_row(self, row: Dict):
        for f in self.fields:
            f.accumulate_stats(row.get(f.name))



    def _build_dataset(self) -> tf.data.Dataset:
        def db_generator():
            for row_dict in self.row_generator():
                try:
                    # Transform input features
                    x = tuple(f.transform(row_dict.get(f.name)) for f in self.fields)
                    # Transform target features (might be different if using flags)
                    # For basic AE, target is same as input
                    y = tuple(f.transform_target(row_dict.get(f.name)) for f in self.fields)
                    # Adjust y if transform_target returns tuples (value, flag)
                    # This basic AE assumes transform_target returns single tensors matching input
                    yield x, x # Target is same as input for simple AE
                except Exception as e:
                    logging.warning(f"Skipping row due to error in generator: {e}. Row: {row_dict}", exc_info=False)
                    continue # Skip rows with errors

        specs_in = tuple(tf.TensorSpec(shape=f.input_shape, dtype=f.input_dtype) for f in self.fields)
        # Target specs should match the output of the generator's y value
        specs_out = tuple(tf.TensorSpec(shape=f.output_shape, dtype=f.output_dtype) for f in self.fields)
        # Adjust specs_out if using optional fields with flags

        ds = tf.data.Dataset.from_generator(db_generator, output_signature=(specs_in, specs_out))

        batch_size = self.config.get("batch_size", 32) # Default batch size
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE) # Add prefetching


    def transform_row(self, row: Dict) -> Dict[str, tf.Tensor]:
        out: Dict[str, tf.Tensor] = {}
        for f in self.fields:
            out[f.name] = f.transform(row.get(f.name))
        return out

    def reconstruct_row(self, latent_vector: np.ndarray) -> Dict[str, str]:
        """
        Decodes a latent vector and converts output tensors to strings by calling
        the field's specific to_string method with the raw decoder output.

        NOTE: This requires field `to_string` methods to handle raw decoder
              output (probabilities/logits) and perform necessary processing
              like argmax internally.
        """
        if self.decoder is None:
            raise RuntimeError("Decoder model is not loaded or assigned to self.decoder.")
        if not self.stats_accumulated:
            raise RuntimeError("Field stats must be finalized before reconstruction.")

        # Ensure latent_vector is 2D (batch dimension of 1)
        if latent_vector.ndim == 1:
             latent_batch = np.expand_dims(latent_vector, axis=0)
        elif latent_vector.ndim == 2 and latent_vector.shape[0] == 1:
             latent_batch = latent_vector
        else:
             raise ValueError(f"Expected latent_vector to be 1D or 2D with batch size 1, got shape {latent_vector.shape}")

        # Predict using the decoder
        # verbose=0 prevents Keras from printing prediction progress for single items
        raw_decoder_outputs = self.decoder.predict(latent_batch, verbose=0)

        decoder_outputs_map = {}
        actual_output_names = []
        if hasattr(self.decoder, 'output_names'):
             # Clean names (remove potential '/...' suffixes added by TF internal ops)
             actual_output_names = [name.split('/')[0] for name in self.decoder.output_names]
        else:
             logging.warning("Decoder model object lacks 'output_names' attribute. Mapping may be unreliable.")

        # --- Logic to handle different predict() output formats ---
        if isinstance(raw_decoder_outputs, dict):
            # Assumes keys are already the correct base names
            # Unbatch each tensor (get the first item)
            decoder_outputs_map = {k: v[0] for k, v in raw_decoder_outputs.items()}

        elif isinstance(raw_decoder_outputs, list):
            if actual_output_names and len(raw_decoder_outputs) == len(actual_output_names):
                # Map using the cleaned output names from the decoder model
                decoder_outputs_map = {name: tensor[0] for name, tensor in zip(actual_output_names, raw_decoder_outputs)} # Unbatch
            elif len(raw_decoder_outputs) == len(self.fields):
                 # Fallback: Map by order if lengths match (less reliable)
                 logging.warning("Decoder output names mismatched or unavailable. Mapping outputs to fields by order.")
                 # Use field.name for keys in this fallback scenario
                 decoder_outputs_map = {field.name: raw_decoder_outputs[i][0] for i, field in enumerate(self.fields)} # Unbatch
            else:
                 raise ValueError(f"Decoder predict list len {len(raw_decoder_outputs)} doesn't match number of fields ({len(self.fields)}) or reliable output names.")

        elif isinstance(raw_decoder_outputs, np.ndarray): # Handle single output case
             if len(self.fields) == 1:
                 # Try to get the actual output name, fallback to field name
                 output_name = actual_output_names[0] if actual_output_names else self.fields[0].name
                 decoder_outputs_map = {output_name: raw_decoder_outputs[0]} # Unbatch
                 logging.info(f"Decoder predict returned single ndarray for single field '{output_name}'. Unbatched.")
             else:
                 raise TypeError(f"Decoder predict returned single ndarray, but multiple fields exist ({len(self.fields)}).")
        else:
            raise TypeError(f"Decoder predict returned unexpected type: {type(raw_decoder_outputs)}")
        # --- End of handling logic ---

        reconstructed_data = {}

        for field in self.fields:
            # Determine the expected output name (consistent with build_autoencoder)
            # Also check just field.name as a fallback if mapping by order occurred
            # expected_output_name_primary = f"{field.name}_decoder"
            expected_output_name_primary = f"{field.name}_recon"
            expected_output_name_fallback = field.name

            output_tensor = decoder_outputs_map.get(expected_output_name_primary)
            used_name = expected_output_name_primary

            if output_tensor is None:
                 output_tensor = decoder_outputs_map.get(expected_output_name_fallback)
                 if output_tensor is not None:
                     used_name = expected_output_name_fallback
                 else:
                     logging.warning(f"Output tensor for field '{field.name}' not found in decoder outputs using expected names ('{expected_output_name_primary}', '{expected_output_name_fallback}'). Skipping. Available keys: {list(decoder_outputs_map.keys())}", exc_info=True)
                     
                     reconstructed_data[field.name] = "[Output Tensor Not Found]"
                     continue # Skip to the next field

            try:
                # Ensure the tensor passed is a numpy array
                if not isinstance(output_tensor, np.ndarray):
                     output_tensor = np.array(output_tensor)

                reconstructed_data[field.name] = field.to_string(output_tensor)

            except ValueError as ve:
                # Catch specific errors like mismatch in lengths if raised by to_string
                logging.error(f"ValueError during to_string for field '{field.name}': {ve}", exc_info=True)
                reconstructed_data[field.name] = f"[ValueError: {ve}]"
            except NotImplementedError:
                 logging.error(f"Field '{field.name}' (type: {type(field).__name__}) has not implemented the required to_string method or called super().", exc_info=True)
                 reconstructed_data[field.name] = "[to_string Not Implemented]"
            except Exception as e:
                # Catch any other unexpected errors during the field's to_string
                logging.error(f"Unexpected error during to_string for field '{field.name}': {e}", exc_info=True)
                reconstructed_data[field.name] = "[Conversion Error]"

        return reconstructed_data

    
    def get_loss_dict(self) -> dict[str, tf.keras.losses.Loss]:
        if self.model is None:
            raise RuntimeError("build_autoencoder() hasn’t been called yet")

        return {
            f"{f.name}_recon": f.loss
            for f in self.fields
        }

    def get_loss_weights_dict(self) -> dict[str, float]:
        if self.model is None:
            raise RuntimeError("build_autoencoder() hasn’t been called yet")

        return {
            f"{f.name}_recon": f.weight
            for f in self.fields
        }



    def print_stats(self):
        print(f"\n--- Stats for {self.__class__.__name__} ---")
        for f in self.fields:
            f.print_stats()
            print("-" * 20) # Separator

    def row_generator(self):
        raise NotImplementedError("Subclasses must implement this method")

    def build_fields(self) -> List["BaseField"]:
        raise NotImplementedError("Subclasses must implement this method")

    def build_autoencoder(self) -> tf.keras.Model:
        if not self.stats_accumulated:
            raise RuntimeError("Call accumulate_stats() / finalize_stats() first")

        latent_dim = self.latent_dim

        # ---------- inputs ----------
        enc_inputs = [
            tf.keras.Input(
                shape=f.input_shape,
                name=f"{f.name}_input",
                dtype=f.input_dtype,
            )
            for f in self.fields
        ]

        # ---------- per‑field encoders / decoders ----------
        enc_outs, decoders = [], {}
        for f, x_in in zip(self.fields, enc_inputs):
            enc_outs.append(f.build_encoder(latent_dim)(x_in))
            decoders[f.name] = f.build_decoder(latent_dim)

        # ---------- fuse latents ----------
        z = enc_outs[0] if len(enc_outs) == 1 else _attention_fuse(
            enc_outs,
            latent_dim=latent_dim,
            num_heads=4,
            key_dim=max(8, latent_dim // 8),
        )
        self.encoder = tf.keras.Model(enc_inputs, z, name="Encoder")

        # ---------- field decoders ----------
        latent_in = tf.keras.Input(shape=(latent_dim,), name="decoder_latent")
        recon_outs = []
        for f in self.fields:
            raw = decoders[f.name](latent_in)
            raw = raw[0] if isinstance(raw, (list, tuple)) else raw

            # wrap in a uniquely‑named linear layer so the tensor gets that name
            recon = tf.keras.layers.Activation(
                "linear", name=f"{f.name}_recon"
            )(raw)
            recon_outs.append(recon)

        self.decoder = tf.keras.Model(latent_in, recon_outs, name="Decoder")

        # ---------- full AE ----------
        full_outs = self.decoder(self.encoder(enc_inputs))
        self.model = tf.keras.Model(enc_inputs, full_outs, name="RowAutoencoder")
        return self.model




    def save_model(self):
        output_dir = Path(self.model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_name_base = self.__class__.__name__
        print(f"Saving models ({model_name_base}) to {output_dir}")
        try:
            if self.model:
                 self.model.save(output_dir / f"{model_name_base}_autoencoder.keras")
            if self.encoder:
                self.encoder.save(output_dir / f"{model_name_base}_encoder.keras")
            if self.decoder:
                self.decoder.save(output_dir / f"{model_name_base}_decoder.keras")
            print("Models saved successfully.")
        except Exception as e:
             print(f"Error saving models: {e}", exc_info=True)

    def load_model(self):
        # Ensure stats are finalized before loading (models depend on field shapes/vocabs)
        if not self.stats_accumulated:
            logging.warning(f"Attempting to load model '{self.__class__.__name__}' before stats are finalized. Finalizing now.")
            # Attempt to finalize. This might fail if accumulation didn't happen.
            try:
                self.finalize_stats()
            except Exception as e:
                 logging.error(f"Failed to finalize stats during load_model: {e}. Model loading might fail.")
                 # Decide whether to proceed or raise error
                 # raise RuntimeError("Cannot load model without finalized stats.") from e

        # Enable deserialization if custom objects are used (like BatchRepel)
        tf.keras.config.enable_unsafe_deserialization()
        input_dir = Path(self.model_dir)
        model_name_base = self.__class__.__name__
        logging.info(f"Loading models ({model_name_base}) from {input_dir}")

        custom_objects = {
            "add_positional_encoding": add_positional_encoding,
            "MaskedSparseCategoricalCrossentropy": MaskedSparseCategoricalCrossentropy,
             "BatchRepel": BatchRepel,
            # Add any other custom layers or functions used in fields.py or here
            # e.g., 'SinActivation': sin_activation if you registered it
        }
        # Include field-specific custom objects if they exist
        for field in self.fields:
             if hasattr(field, 'get_custom_objects'):
                  custom_objects.update(field.get_custom_objects())


        try:
            # Load the main autoencoder first to reconstruct the architecture
            self.model = tf.keras.models.load_model(
                 input_dir / f"{model_name_base}_autoencoder.keras",
                 custom_objects=custom_objects,
                 compile=False # Usually recompile before training
            )
            # Load encoder and decoder separately
            self.encoder = tf.keras.models.load_model(
                input_dir / f"{model_name_base}_encoder.keras",
                custom_objects=custom_objects,
                 compile=False
             )
            self.decoder = tf.keras.models.load_model(
                 input_dir / f"{model_name_base}_decoder.keras",
                 custom_objects=custom_objects,
                 compile=False
            )
            logging.info(f"Models ({model_name_base}) loaded successfully.")
            # Optional: Print summary after loading
            # self._print_model_architecture()

        except Exception as e:
            logging.error(f"Error loading model files for {model_name_base} from {input_dir}: {e}", exc_info=True)
            # Set models to None or raise the error depending on desired behavior
            self.model = None
            self.encoder = None
            self.decoder = None
            raise FileNotFoundError(f"Model files not found or failed to load in {input_dir} for {model_name_base}") from e


    def _print_model_architecture(self):
        if self.model:
            print("\n--- Main Autoencoder Summary ---")
            self.model.summary(line_length=120)
            print("Autoencoder Outputs:", self.model.output_names)
        else:
            print("\n--- Main Autoencoder not built or loaded yet ---")

        if self.encoder:
            print("\n--- Encoder Summary ---")
            self.encoder.summary(line_length=120)
        else:
             print("\n--- Encoder not built or loaded yet ---")

        if self.decoder:
            print("\n--- Decoder Summary ---")
            self.decoder.summary(line_length=120)
            print("Decoder Outputs:", self.decoder.output_names)

        else:
            print("\n--- Decoder not built or loaded yet ---")

        print("\n--- Field Specific Model Summaries ---")
        for field in self.fields:
             try:
                print(f"\n--- {field.name} Encoder ---")
                field.build_encoder(self.latent_dim).summary()
                print(f"\n--- {field.name} Decoder ---")
                field.build_decoder(self.latent_dim).summary()
             except Exception as e:
                print(f"Could not build/summarize models for field {field.name}: {e}")


    def fit(self):
        if not self.stats_accumulated:
            self.accumulate_stats()
            self.finalize_stats()

        if self.model is None:
            self.build_autoencoder()

        self._print_model_architecture()

        epochs          = self.config.get("epochs", 10)
        initial_lr      = self.config.get("learning_rate", 2e-4)
        weight_decay    = self.config.get("weight_decay", 1e-4)
        callback_every  = self.config.get("callback_interval", 100)

        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=initial_lr,
            weight_decay=weight_decay,
        )

        self.model.compile(
            optimizer=optimizer,
            loss=[f.loss for f in self.fields],
            loss_weights=[f.weight for f in self.fields],
        )

        log_root = Path(self.config["log_dir"])
        log_dir  = log_root / "fit" / self.__class__.__name__
        log_dir.mkdir(parents=True, exist_ok=True)

        tensorboard_callback = TensorBoardPerBatchLoggingCallback(
            log_dir=log_dir,
            log_interval=callback_every,
        )

        reconstruction_callback = ReconstructionCallback(
            interval_batches=callback_every,
            row_autoencoder=self,
            db_path=self.db_path,
            num_samples=5,
        )

        model_save_callback = ModelSaveCallback(self, output_dir=self.model_dir)

        ds = self._build_dataset()

        steps_per_epoch = (
            self.num_rows_in_dataset // self.config.get("batch_size", 32)
            if self.num_rows_in_dataset else None
        )

        history = self.model.fit(
            ds,
            epochs=epochs,
            callbacks=[
                tensorboard_callback,
                reconstruction_callback,
                model_save_callback,
                tf.keras.callbacks.TerminateOnNaN(),
            ],
            steps_per_epoch=steps_per_epoch,
            verbose=1,
        )

        self.save_model()
        return history
