# schema.py
from pathlib import Path
from prettytable import PrettyTable
from typing import Any, List, Dict

from tqdm import tqdm

from .fields import BaseField, MaskedSparseCategoricalCrossentropy, add_positional_encoding, sin_activation
import tensorflow as tf
from functools import cached_property


class RowAutoencoder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fields = self.build_fields()
        self.num_rows_in_dataset = 0
        self.latent_dim = self.config["latent_dim"]
        self.stats_accumulated = False

    def accumulate_stats(self):
        
        if self.stats_accumulated:
            print("stats already accumulated")
            return
        
        num_rows = 0

        print("Accumulating stats...")
        for row in tqdm(self.row_generator(), desc="Accumulating stats"):
            self.accumulate_stats_for_row(row)
            num_rows += 1

        self.num_rows_in_dataset = num_rows
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

    def _build_dataset(self) -> tf.data.Dataset:
        def db_generator():
             for row_dict in self.row_generator():
                x = tuple(f.transform(row_dict.get(f.name)) for f in self.fields)
                yield x, x

        # Define the signature for tf.data
        specs_in = tuple(tf.TensorSpec(shape=f.input_shape, dtype=f.input_dtype) for f in self.fields)
        specs_out = tuple(tf.TensorSpec(shape=f.output_shape, dtype=tf.float32) for f in self.fields)

        ds = tf.data.Dataset.from_generator(db_generator, output_signature=(specs_in, specs_out))
        return ds.batch(self.config["batch_size"])

    def transform_row(self, row: Dict) -> Dict[str, tf.Tensor]:
        out = {}
        for f in self.fields:
            raw_value = row.get(f.name, None)
            out[f.name] = f.transform(raw_value)
        return out

    def get_loss_dict(self) -> Dict[str, str]:
        return {f"{f.name}_decoder": f.loss for f in self.fields}

    def get_loss_weights_dict(self) -> Dict[str, float]:
        return {f"{f.name}_decoder": f.weight for f in self.fields}

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
        latent_dim = self.latent_dim
        encoder_inputs = {}
        encoder_outputs = {}
        decoders = {}
        decoder_outputs = {}

        # Build inputs and per-field encoders/decoders.
        for field in self.fields:
            inp = tf.keras.Input(shape=field.input_shape, name=f"{field.name}_input", dtype=field.input_dtype)
            encoder_inputs[field.name] = inp

        for field in self.fields:
            encoder = field.build_encoder(latent_dim)
            decoder = field.build_decoder(latent_dim)
            decoders[field.name] = decoder
            encoder_outputs[field.name] = encoder(encoder_inputs[field.name])

        combined_encodings = tf.keras.layers.Concatenate()(list(encoder_outputs.values()))
        latent_vector = tf.keras.layers.Dense(latent_dim * 8, activation='gelu', name="latent_vector_dense1")(combined_encodings)
        latent_vector = tf.keras.layers.Dense(latent_dim * 2, activation='gelu', name="latent_vector_dense2")(latent_vector)
        latent_vector = tf.keras.layers.Dense(latent_dim, activation='linear', name="latent_vector_dense3")(latent_vector)
        latent_vector = tf.keras.layers.LayerNormalization(name="latent_vector_layernorm")(latent_vector)

        # Decode the latent vector.
        for field in self.fields:
            decoder_outputs[field.name] = decoders[field.name](latent_vector)

        autoencoder = tf.keras.Model(
            inputs=[encoder_inputs[field.name] for field in self.fields],
            outputs=[decoder_outputs[field.name] for field in self.fields],
            name="TabularAutoencoder"
        )

        # Create an encoder model (from input layers to the latent vector).
        encoder_model = tf.keras.Model(
            inputs=[encoder_inputs[field.name] for field in self.fields],
            outputs=latent_vector,
            name="Encoder"
        )

        # Create a decoder model (from a latent vector input to decoder outputs).
        latent_input = tf.keras.Input(shape=(latent_dim,), name="latent_input")
        decoder_outputs_for_decoder = [decoders[field.name](latent_input) for field in self.fields]
        decoder_model = tf.keras.Model(
            inputs=latent_input,
            outputs=decoder_outputs_for_decoder,
            name="Decoder"
        )

        # Store the models.
        self.model = autoencoder
        self.encoder = encoder_model
        self.decoder = decoder_model
        return autoencoder

    def save_model(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(output_dir / f"{self.__class__.__name__}_autoencoder.keras")
        self.encoder.save(output_dir / f"{self.__class__.__name__}_encoder.keras")
        self.decoder.save(output_dir / f"{self.__class__.__name__}_decoder.keras")

    def load_model(self, input_dir):
        tf.keras.config.enable_unsafe_deserialization()
        input_dir = Path(input_dir)
        try:
            custom_objects = {"add_positional_encoding": add_positional_encoding, "sin_activation": sin_activation, "MaskedSparseCategoricalCrossentropy": MaskedSparseCategoricalCrossentropy}

            self.model = tf.keras.models.load_model(input_dir / f"{self.__class__.__name__}_autoencoder.keras", custom_objects=custom_objects)
            self.encoder = tf.keras.models.load_model(input_dir / f"{self.__class__.__name__}_encoder.keras", custom_objects=custom_objects)
            self.decoder = tf.keras.models.load_model(input_dir / f"{self.__class__.__name__}_decoder.keras", custom_objects=custom_objects)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model files not found in {input_dir}")


class TableJoinSequenceEncoder:
    def __init__(self, db_path, first_table_schema, second_table_schema, config, name=None):
        self.first_table_schema = first_table_schema
        self.second_table_schema = second_table_schema
        self.db_path = db_path
        self.config = config
        self.stats_accumulated = False

        self.name = name or self.__class__.__name__

    def accumulate_stats(self):
        if self.stats_accumulated:
            print("stats already accumulated")
            return
        self.first_table_schema.accumulate_stats()
        self.second_table_schema.accumulate_stats()
        self.stats_accumulated = True

    def row_generator(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    def build_model(self):
        latent_dim = self.config["latent_dim"]
        max_seq_len = self.config.get("max_seq_len", 10)

        # Build the movie (input table) branch.
        first_table_inputs = {}
        first_table_encodings = {}
        first_table_decoders = {}
        for field in self.first_table_schema.fields:
            inp = tf.keras.Input(shape=field.input_shape, name=f"{field.name}_input")
            first_table_inputs[field.name] = inp
            encoder = field.build_encoder(latent_dim)
            first_table_encodings[field.name] = encoder(inp)
            decoder = field.build_decoder(latent_dim)
            first_table_decoders[field.name] = decoder

        combined_encoding = tf.keras.layers.Concatenate()(list(first_table_encodings.values()))
        latent = tf.keras.layers.Dense(latent_dim, activation='linear')(combined_encoding)
        latent = tf.keras.layers.LayerNormalization()(latent)

        # Movie reconstruction branch (ensures the input representation is learned properly)
        movie_outputs = {}
        for field in self.first_table_schema.fields:
            movie_outputs[f"{field.name}_decoder"] = first_table_decoders[field.name](latent)

        # Credit sequence branch. The latent vector is repeated and fed through an LSTM.
        repeated_latent = tf.keras.layers.RepeatVector(max_seq_len)(latent)
        credit_lstm = tf.keras.layers.LSTM(latent_dim, return_sequences=True, name="credit_lstm")(repeated_latent)
        credit_outputs = {}
        for field in self.second_table_schema.fields:
            credit_decoder = field.build_decoder(latent_dim)
            # Apply the credit decoder at each time step.
            td = tf.keras.layers.TimeDistributed(credit_decoder, name=f"{field.name}_seq_decoder")
            credit_outputs[f"{field.name}_seq_decoder"] = td(credit_lstm)

        inputs = list(first_table_inputs.values())
        outputs = list(movie_outputs.values()) + list(credit_outputs.values())

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="MovieCreditSequenceModel")
        return model
    
    