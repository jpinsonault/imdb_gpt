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
        self.num_rows_in_dataset = 0

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


class UncertaintyWeightedAutoencoder(tf.keras.Model):
    def __init__(self, row_autoencoder: RowAutoencoder, **kwargs):
        super().__init__(**kwargs)
        self.row_autoencoder = row_autoencoder
        self.autoencoder = row_autoencoder.build_autoencoder()
        self.uncertainties = {}
        for field in self.row_autoencoder.fields:
            key = f"{field.name}_decoder"
            self.uncertainties[key] = tf.Variable(
                0.0, trainable=True, dtype=tf.float32, name=f"{key}_uncertainty"
            )

    def call(self, inputs, training=False):
        return self.autoencoder(inputs, training=training)

    def train_step(self, data):
        inputs, targets = data
        with tf.GradientTape() as tape:
            outputs = self(inputs, training=True)
            total_loss = 0.0
            loss_dict = self.row_autoencoder.get_loss_dict()
            field_losses = {}

            for i, field in enumerate(self.row_autoencoder.fields):
                key = f"{field.name}_decoder"
                loss_fn = loss_dict[key]
                y_true = targets[i]
                y_pred = outputs[i]

                raw_loss = loss_fn(y_true, y_pred)
                s = self.uncertainties[key]
                weighted_loss = tf.exp(-s) * raw_loss + s
                field_losses[key] = weighted_loss
                total_loss += weighted_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        metrics = {"loss": total_loss}
        metrics.update(field_losses)
        return metrics
    

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
    
    