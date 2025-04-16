import logging
import numpy as np
from pathlib import Path
from prettytable import PrettyTable
from typing import Any, List, Dict

from tqdm import tqdm
import tensorflow as tf
from scripts.autoencoder.fields import BaseField, MaskedSparseCategoricalCrossentropy, add_positional_encoding, digit_loss
import tensorflow_probability as tfp

# Alias for distributions
tfd = tfp.distributions
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

################################################################################
# RowAutoencoder (Updated to use the custom latent layer)
################################################################################
class RowAutoencoder:
    def __init__(self, config: Dict[str, Any], model_dir: Path):
        self.config = config
        self.model_dir = model_dir
        self.fields = self.build_fields()
        self.num_rows_in_dataset = 0
        self.latent_dim = self.config["latent_dim"]
        self.stats_accumulated = False

    def accumulate_stats(self):
        if self.stats_accumulated:
            logging.info("Stats already accumulated.")
            return

        num_rows = 0
        logging.info("Starting accumulation of stats.")
        for row in tqdm(self.row_generator(), desc="Accumulating stats"):
            self.accumulate_stats_for_row(row)
            num_rows += 1
        self.num_rows_in_dataset = num_rows
        self.finalize_stats()
        self.print_stats()
        self.stats_accumulated = True
        logging.info(f"Finished accumulating stats for {num_rows} rows.")

    def accumulate_stats_for_row(self, row: Dict):
        for f in self.fields:
            raw_value = row.get(f.name, None)
            f.accumulate_stats(raw_value)

    def finalize_stats(self):
        for f in self.fields:
            f.finalize_stats()

    def _build_dataset(self) -> tf.data.Dataset:
        def db_generator():
            for row_dict in self.row_generator():
                x = tuple(f.transform(row_dict.get(f.name)) for f in self.fields)
                yield x, x

        specs_in = tuple(tf.TensorSpec(shape=f.input_shape, dtype=f.input_dtype) for f in self.fields)
        specs_out = tuple(tf.TensorSpec(shape=f.output_shape, dtype=f.output_dtype) for f in self.fields)

        ds = tf.data.Dataset.from_generator(db_generator, output_signature=(specs_in, specs_out))
        return ds.batch(self.config["batch_size"])


    def transform_row(self, row: Dict) -> Dict[str, tf.Tensor]:
        out = {}
        for f in self.fields:
            raw_value = row.get(f.name, None)
            out[f.name] = f.transform(raw_value)
        return out

    def get_loss_dict(self) -> Dict[str, Any]:
        return {f"{f.name}_decoder": f.loss for f in self.fields}

    def get_loss_weights_dict(self) -> Dict[str, float]:
        if len(self.fields) == 1:
            return {f"{self.fields[0].name}_decoder": self.fields[0].weight}
        return {f"{f.name}_decoder": f.weight for f in self.fields}

    def print_stats(self):
        for f in self.fields:
            f.print_stats()
            print()

    def row_generator(self):
        raise NotImplementedError("Subclasses must implement this method")

    def build_fields(self) -> List["BaseField"]:
        raise NotImplementedError("Subclasses must implement this method")

    ############################################################################
    # Updated build_autoencoder that uses the custom LatentLayer
    ############################################################################
    def build_autoencoder(self) -> tf.keras.Model:
        latent_dim = self.latent_dim
        inputs = {}
        encoder_outputs = {}
        decoders = {}
        decoder_outputs = {}

        # Create input layers for each field.
        for field in self.fields:
            input_tensor = tf.keras.Input(
                shape=field.input_shape, name=f"{field.name}_input", dtype=field.input_dtype
            )
            inputs[field.name] = input_tensor

        # Build the encoders for each field.
        for field in self.fields:
            encoder = field.build_encoder(latent_dim)
            decoder = field.build_decoder(latent_dim)
            decoders[f"{field.name}_decoder"] = decoder
            encoder_outputs[field.name] = encoder(inputs[field.name])

        # Concatenate all encoder outputs.
        combined_encodings = tf.keras.layers.Concatenate()(list(encoder_outputs.values()))
        x = tf.keras.layers.Dense(latent_dim, activation='gelu', name="dense_bottleneck_1")(combined_encodings)
        # Produce latent parameters (concatenated mean and log variance).
        x = tf.keras.layers.Dense(latent_dim * 2, name="latent_params")(x)

        latent_vector = tf.keras.layers.Dense(latent_dim, activation='gelu', name="dense_bottleneck_2")(x)
        latent_vector = tf.keras.layers.LayerNormalization(name="latent_layernorm")(latent_vector)

        # Build each decoder branch.
        for field in self.fields:
            decoder_outputs[f"{field.name}_decoder"] = decoders[f"{field.name}_decoder"](latent_vector)

        autoencoder = tf.keras.Model(
            inputs=[inputs[field.name] for field in self.fields],
            outputs=[decoder_outputs[f"{field.name}_decoder"] for field in self.fields],
            name="TabularDenoisingAutoencoder",
        )

        # Create separate encoder and decoder models.
        encoder_model = tf.keras.Model(
            inputs=[inputs[field.name] for field in self.fields],
            outputs=latent_vector,
            name="Encoder",
        )

        latent_input = tf.keras.Input(shape=(latent_dim,), name="latent_input")
        decoder_outputs_for_decoder = {}
        for field in self.fields:
            reconstruction = decoders[f"{field.name}_decoder"](latent_input)
            decoder_outputs_for_decoder[f"{field.name}_decoder"] = reconstruction

        decoder_model = tf.keras.Model(
            inputs=latent_input,
            outputs=decoder_outputs_for_decoder,
            name="Decoder",
        )

        self.model = autoencoder
        self.encoder = encoder_model
        self.decoder = decoder_model
        return autoencoder

    def save_model(self):
        output_dir = Path(self.model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving model to {output_dir}")
        self.model.save(output_dir / f"{self.__class__.__name__}_autoencoder.keras")
        self.encoder.save(output_dir / f"{self.__class__.__name__}_encoder.keras")
        self.decoder.save(output_dir / f"{self.__class__.__name__}_decoder.keras")

    def load_model(self):
        tf.keras.config.enable_unsafe_deserialization()
        input_dir = Path(self.model_dir)
        try:
            custom_objects = {
                "add_positional_encoding": add_positional_encoding,
                "MaskedSparseCategoricalCrossentropy": MaskedSparseCategoricalCrossentropy,
                "digit_loss": digit_loss,
            }
            self.model = tf.keras.models.load_model(input_dir / f"{self.__class__.__name__}_autoencoder.keras",
                                                      custom_objects=custom_objects)
            self.encoder = tf.keras.models.load_model(input_dir / f"{self.__class__.__name__}_encoder.keras",
                                                        custom_objects=custom_objects)
            self.decoder = tf.keras.models.load_model(input_dir / f"{self.__class__.__name__}_decoder.keras",
                                                        custom_objects=custom_objects)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model files not found in {input_dir}")
