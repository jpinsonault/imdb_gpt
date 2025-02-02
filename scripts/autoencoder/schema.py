# schema.py
from prettytable import PrettyTable
from typing import List, Dict

from tqdm import tqdm

from .fields import BaseField
import tensorflow as tf
from functools import cached_property

class RowAutoencoder:
    def __init__(self):
        self.fields = self.build_fields()

    def accumulate_stats(self):
        if self.stats_accumulated:
            print("stats already accumulated")
            return

        print("Accumulating stats...")
        for row in tqdm(self.row_generator(), desc="Accumulating stats"):
            self.accumulate_stats_for_row(row)

        self.finalize_stats()
        self.print_stats()
        self.stats_accumulated = True

    def accumulate_stats_for_row(self, row: Dict):
        for f in self.fields:
            raw_value = row.get(f.name, None)
            f.accumulate_stats(raw_value)

    def finalize_stats(self):
        for f in self.fields:
            f.finalize_stats()

    def transform_row(self, row: Dict) -> Dict[str, tf.Tensor]:
        out = {}
        for f in self.fields:
            raw_value = row.get(f.name, None)
            out[f.name] = f.transform(raw_value)
        return out

    def get_loss_dict(self) -> Dict[str, str]:
        return {f"{f.name}_decoder": f.loss for f in self.fields}

    def get_loss_weights_dict(self) -> Dict[str, float]:
        return {f"{f.name}_decoder": 1.0 for f in self.fields}

    def print_stats(self):
        for f in self.fields:
            f.print_stats()
            print()

    def row_generator(self):
        raise NotImplementedError("Subclasses must implement this method")

    def build_fields(self) -> List[BaseField]:
        raise NotImplementedError("Subclasses must implement this method")

    def _build_model(self):
        if not self.model:
            self.model = self.build_autoencoder()

    def build_autoencoder(self) -> tf.keras.Model:
        latent_dim = self.config["latent_dim"]
        encoder_inputs = {}
        encoder_outputs = {}
        decoders = {}
        decoder_outputs = {}

        # Create input layers
        for field in self.fields:
            input_layer = tf.keras.Input(shape=field.input_shape, name=f"{field.name}_input", dtype=field.input_dtype)
            encoder_inputs[field.name] = input_layer

        # Create encoders and connect them to the corresponding input layers
        for field in self.fields:
            encoder = field.build_encoder(latent_dim)
            decoder = field.build_decoder(latent_dim)

            decoders[field.name] = decoder

            encoder_output = encoder(encoder_inputs[field.name])
            encoder_outputs[field.name] = encoder_output

        combined_encodings = tf.keras.layers.Concatenate()(list(encoder_outputs.values()))

        latent_vector = tf.keras.layers.Dense(latent_dim*8, activation='gelu', name="latent_vector_dense1")(combined_encodings)
        latent_vector = tf.keras.layers.Dense(latent_dim*2, activation='gelu', name="latent_vector_dense2")(latent_vector)

        latent_vector = tf.keras.layers.Dense(latent_dim, activation='linear', name="latent_vector_dense3")(latent_vector)
        latent_vector = tf.keras.layers.LayerNormalization()(latent_vector)

        # Decode the latent vector
        for field in self.fields:
            decoder = decoders[field.name]
            decoder.name = f"{field.name}_decoder"
            decoder_outputs[field.name] = decoder(latent_vector)

        # Create the autoencoder model, ensuring the input order matches the schema
        input_layers = [encoder_inputs[field.name] for field in self.fields]
        output_layers = [decoder_outputs[field.name] for field in self.fields]

        autoencoder = tf.keras.Model(
            inputs=input_layers,
            outputs=output_layers,
            name="TabularAutoencoder"
        )
        return autoencoder


class TableJoinSequenceEncoder:
    def __init__(self, db_path, from_table_schema: RowAutoencoder, to_table_schema: RowAutoencoder):
        self.from_table_schema = from_table_schema
        self.to_table_schema = to_table_schema
        self.db_path = db_path
        self.stats_accumulated = False

    def accumulate_stats(self):
        if self.stats_accumulated:
            print("stats already accumulated")
            return

        self.from_table_schema.accumulate_stats()
        self.to_table_schema.accumulate_stats()

        self.stats_accumulated = True

    def row_generator(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    def build_model(self):
        # lstm
        pass