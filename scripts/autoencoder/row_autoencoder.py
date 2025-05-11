import logging
import numpy as np
from pathlib import Path
from prettytable import PrettyTable
from typing import Any, List, Dict
import os
from tqdm import tqdm
import tensorflow as tf
from autoencoder.fields import BaseField, BooleanField, MaskedSparseCategoricalCrossentropy, MultiCategoryField, NumericDigitCategoryField, ScalarField, SingleCategoryField, add_positional_encoding, TextField
from autoencoder.training_callbacks import ModelSaveCallback, ReconstructionCallback

# Alias for distributions
logging.basicConfig(level=logging.info, format='%(asctime)s - %(levelname)s - %(message)s')

    
@tf.keras.utils.register_keras_serializable(package="Custom", name="BatchRepel")
class BatchRepel(tf.keras.layers.Layer):
    def __init__(self, coeff=1e-2, **kw):
        super().__init__(**kw)
        self.coeff = coeff

    def call(self, z):
        # unit‑length so that cosine == dot product
        z = tf.nn.l2_normalize(z, axis=-1)

        sim = tf.matmul(z, z, transpose_b=True)          # (B,B)
        b = tf.shape(sim)[0]
        mask = tf.ones_like(sim) - tf.eye(b)             # zero out self‑similarities
        off_diag = sim * mask
        reg = tf.reduce_sum(tf.square(off_diag)) / tf.cast(b * (b - 1), tf.float32)

        self.add_loss(self.coeff * reg)
        return z
    
    def get_config(self):
        config = super().get_config()
        config.update({"coeff": self.coeff})
        return config


################################################################################
# RowAutoencoder (Updated to use the custom latent layer)
################################################################################
class RowAutoencoder:
    def __init__(self, config: Dict[str, Any], model_dir: Path):
        self.config = config
        self.model: Optional[tf.keras.Model] = None
        self.encoder: Optional[tf.keras.Model] = None
        self.decoder: Optional[tf.keras.Model] = None
        self.model_dir = Path(model_dir) # Ensure it's a Path object
        self.fields = self.build_fields()
        self.num_rows_in_dataset = 0
        self.latent_dim = self.config["latent_dim"]
        self.stats_accumulated = False
        # Added db_path reference needed for fit method's callback
        self.db_path = config['db_path'] # Get db_path from config if passed

    def accumulate_stats(self):
        if self.stats_accumulated:
            logging.info("Stats already accumulated.")
            return

        num_rows = 0
        logging.info(f"Starting accumulation of stats for {self.__class__.__name__}.")
        # Wrap row_generator with tqdm for progress bar
        for row in tqdm(self.row_generator(), desc=f"Accumulating stats ({self.__class__.__name__})"):
            self.accumulate_stats_for_row(row)
            num_rows += 1
        self.num_rows_in_dataset = num_rows
        logging.info(f"Finished accumulating stats for {num_rows} rows for {self.__class__.__name__}.")

    def accumulate_stats_for_row(self, row: Dict):
        for f in self.fields:
            raw_value = row.get(f.name, None)
            f.accumulate_stats(raw_value)

    def finalize_stats(self):
        logging.info(f"Finalizing stats for {self.__class__.__name__}...")
        for f in self.fields:
            f.finalize_stats()
        self.stats_accumulated = True # Set flag here
        logging.info(f"Stats finalized for {self.__class__.__name__}.")


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
        """Transforms a dictionary row into a dictionary of tensors for encoder input."""
        out = {}
        for f in self.fields:
            raw_value = row.get(f.name, None)
            try:
                 # Ensure field stats are finalized before transforming
                if not f.stats_finalized(): # Add a check method to BaseField if needed
                     raise RuntimeError(f"Stats for field '{f.name}' must be finalized before transforming.")
                out[f.name] = f.transform(raw_value)
            except Exception as e:
                logging.error(f"Error transforming field '{f.name}' with value '{raw_value}': {e}", exc_info=True)
                # Handle error appropriately, e.g., return default padding or raise
                # For now, let's try to return padding, assuming get_base_padding_value works
                try:
                    out[f.name] = f.get_base_padding_value()
                    logging.warning(f"Using padding value for field '{f.name}' due to transformation error.")
                except Exception as pad_e:
                     logging.error(f"Could not get padding value for field '{f.name}': {pad_e}")
                     raise e # Re-raise original error if padding fails
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
            logging.info("Decoder predict returned dict. Unbatched.")

        elif isinstance(raw_decoder_outputs, list):
            if actual_output_names and len(raw_decoder_outputs) == len(actual_output_names):
                # Map using the cleaned output names from the decoder model
                decoder_outputs_map = {name: tensor[0] for name, tensor in zip(actual_output_names, raw_decoder_outputs)} # Unbatch
                logging.info(f"Decoder predict returned list matching output_names {actual_output_names}. Unbatched.")
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
        logging.info(f"Decoder map keys after processing predict() output: {list(decoder_outputs_map.keys())}")

        for field in self.fields:
            # Determine the expected output name (consistent with build_autoencoder)
            # Also check just field.name as a fallback if mapping by order occurred
            expected_output_name_primary = f"{field.name}_decoder"
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
                # The core change: Pass the raw, unbatched tensor directly.
                # The field's to_string method must handle this raw tensor.
                logging.info(f"Calling to_string for field '{field.name}' (type: {type(field).__name__}) using output key '{used_name}' with tensor shape: {output_tensor.shape}")

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

    
    def get_loss_dict(self) -> Dict[str, Any]:
        if self.model is None or not hasattr(self.model, "output_names"):
            raise RuntimeError("Model must be built (and assigned to self.model) before calling get_loss_dict().")
        # Zip the model's actual outputs with the fields in the same order you appended them
        return {
            output_name: field.loss
            for output_name, field in zip(self.model.output_names, self.fields)
        }

    def get_loss_weights_dict(self) -> Dict[str, float]:
        if self.model is None or not hasattr(self.model, "output_names"):
            raise RuntimeError("Model must be built (and assigned to self.model) before calling get_loss_weights_dict().")
        return {
            output_name: field.weight
            for output_name, field in zip(self.model.output_names, self.fields)
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
            raise RuntimeError("Stats must be accumulated and finalized before building the model.")

        latent_dim = self.latent_dim

        # ---------- inputs ----------
        inp_list = []
        for f in self.fields:
            inp = tf.keras.Input(shape=f.input_shape,
                                name=f"{f.name}_input",
                                dtype=f.input_dtype)
            inp_list.append(inp)

        # ---------- encoders ----------
        enc_outs = []
        decoders = {}
        for f, inp in zip(self.fields, inp_list):
            enc = f.build_encoder(latent_dim)
            dec = f.build_decoder(latent_dim)
            enc_outs.append(enc(inp))
            decoders[f.name] = dec             # keep for the combined decoder

        # ---------- latent bottleneck ----------
        if len(enc_outs) == 1:
            z = enc_outs[0]
        else:
            z = tf.keras.layers.Concatenate(name="concat_latent")(enc_outs)

        z = tf.keras.layers.Dense(latent_dim * 2, activation="gelu")(z)
        z = tf.keras.layers.LayerNormalization()(z)
        z = tf.keras.layers.Dense(latent_dim, activation="gelu")(z)
        z = tf.keras.layers.LayerNormalization(name="latent")(z)

        self.encoder = tf.keras.Model(inp_list, z, name="Encoder")

        # ---------- combined decoder (NEW) ----------
        dec_in = tf.keras.Input(shape=(latent_dim,), name="decoder_latent")
        dec_outs = []
        for f in self.fields:
            out = decoders[f.name](dec_in)
            main = out[0] if isinstance(out, (list, tuple)) else out
            dec_outs.append(main)
        self.decoder = tf.keras.Model(dec_in, dec_outs, name="Decoder")

        # ---------- full autoencoder ----------
        ae_outs = self.decoder(z)
        self.model = tf.keras.Model(inp_list, ae_outs, name="RowAutoencoder")
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

        # Optionally print field-specific encoder/decoder summaries if needed
        # print("\n--- Field Specific Model Summaries ---")
        # for field in self.fields:
        #      try:
        #         print(f"\n--- {field.name} Encoder ---")
        #         field.build_encoder(self.latent_dim).summary()
        #         print(f"\n--- {field.name} Decoder ---")
        #         field.build_decoder(self.latent_dim).summary()
        #      except Exception as e:
        #         print(f"Could not build/summarize models for field {field.name}: {e}")


    def fit(self):
        # Ensure stats are ready
        if not self.stats_accumulated:
             print("Stats not accumulated, running accumulation first.")
             self.accumulate_stats()
             self.finalize_stats() # Ensure finalization happens after accumulation

        # Build model if not loaded/built
        if self.model is None:
             print("Model not found, building autoencoder...")
             self.build_autoencoder()
             print("Autoencoder built.")

        # Print architecture
        self._print_model_architecture()

        # --- Training Configuration ---
        epochs = self.config.get("epochs", 10) # Default epochs
        initial_lr = self.config.get("learning_rate", 0.0002)
        weight_decay = self.config.get("weight_decay", 1e-4)
        callback_interval = self.config.get("callback_interval", 100) # Batches

        # Learning Rate Schedule (Example: fixed for simplicity)
        # schedule = [initial_lr] * epochs # Fixed LR
        # More complex schedules can be defined here
        # def scheduler(epoch, lr):
        #     if epoch < 10: return initial_lr
        #     else: return initial_lr * tf.math.exp(-0.1 * (epoch - 9))
        # lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

        # Optimizer
        optimizer = tf.keras.optimizers.AdamW(learning_rate=initial_lr, weight_decay=weight_decay)

        # Compile the model
        try:
            loss_dict = self.get_loss_dict()
            loss_weights_dict = self.get_loss_weights_dict()
            print("\nCompiling model with:")
            print("  Losses:", loss_dict)
            print("  Loss Weights:", loss_weights_dict)
            self.model.compile(
                optimizer=optimizer,
                loss=loss_dict,
                loss_weights=loss_weights_dict,
                 # Add metrics if needed, e.g., ['accuracy'] for categorical outputs
                 # metrics={output_name: ['accuracy'] for output_name in loss_dict}
             )
            print("Model compiled successfully.")
        except Exception as e:
             logging.error(f"Error compiling model: {e}", exc_info=True)
             logging.error(f"Model Output Names: {self.model.output_names if self.model else 'N/A'}")
             logging.error(f"Loss Dict Keys: {list(self.get_loss_dict().keys())}")
             raise RuntimeError("Failed to compile model.") from e


        # Callbacks
        log_dir = "logs/fit/" + self.__class__.__name__ + "_" + str(int(tf.timestamp().numpy()))
        os.makedirs(log_dir, exist_ok=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1, # Log histograms every epoch
             write_graph=True,
             write_images=False,
             update_freq='epoch', # Log metrics each epoch
            # embeddings_freq=1, # Log embeddings if layer names provided
             profile_batch=0 # Disable profiler or set range e.g., '10,20'
        )

        # Ensure db_path is available for ReconstructionCallback
        if not self.db_path:
             # Try to get it from the class if it was set directly
             self.db_path = getattr(self, 'db_path', None)
             if not self.db_path:
                  raise ValueError("db_path is required for ReconstructionCallback but not found in config or class.")


        reconstruction_callback = ReconstructionCallback(
            interval_batches=callback_interval,
            row_autoencoder=self,
            db_path=self.db_path, # Pass db_path here
             num_samples=5 # Reduced sample count for faster callback
        )

        # Model saving callback (saves at end of epochs)
        model_save_callback = ModelSaveCallback(self, output_dir=self.model_dir)

        callbacks_list = [
            tensorboard_callback,
            reconstruction_callback,
             # lr_callback, # Add LR scheduler if using one
            model_save_callback,
            tf.keras.callbacks.TerminateOnNaN() # Stop training if loss becomes NaN
        ]

        # Build the dataset
        print("Building dataset...")
        ds = self._build_dataset()
        print("Dataset built.")

        # Determine steps per epoch if possible
        steps_per_epoch = None
        if self.num_rows_in_dataset > 0 and self.config.get("batch_size", 32) > 0:
             steps_per_epoch = self.num_rows_in_dataset // self.config.get("batch_size", 32)
             print(f"Estimated steps per epoch: {steps_per_epoch}")
        else:
             print("Could not determine steps per epoch (num_rows_in_dataset or batch_size missing/zero).")


        # Train the model
        print(f"\nStarting training for {epochs} epochs...")
        history = self.model.fit(
            ds,
            epochs=epochs,
            callbacks=callbacks_list,
            steps_per_epoch=steps_per_epoch, # Optional: useful with tf.data repeat()
             verbose=1 # Show progress bar and metrics per epoch
        )
        print("\nTraining complete.")

        # Optionally save again explicitly after training finishes
        print("Saving final model...")
        self.save_model()

        return history