from datetime import datetime
from functools import partial
import json
import sqlite3
import sys
import traceback
from typing import List, Tuple
# import opencv
from tqdm import tqdm
from pathlib import Path
import numpy as np
from collections import defaultdict
from scripts.train_imdb_llm import ReduceSum, sinusoidal_encoding
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention, LayerNormalization, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, UpSampling1D, Add, Reshape, BatchNormalization, Layer, Conv1DTranspose, LeakyReLU, DepthwiseConv1D, Conv1DTranspose, GlobalAveragePooling1D, Concatenate, Permute, Multiply, Add, Dense, Flatten, GlobalMaxPooling1D, Layer, Softmax, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from config import project_config
from scripts.attention_model import generative_matrix_attention
from scripts.utils import print_project_config
# from torch.utils.data import Dataset, DataLoader
import csv
import random
import os
import json
import hashlib
from typing import Set, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from track_changes import commit_changes_to_git, suggest_folder_name

tf_version = tf.__version__

SPECIAL_PAD = '\u200C'
SPECIAL_START = '\u200D'
SPECIAL_END = '\u200E'


class TokenProcessor:
    def __init__(self, char_to_index, max_input_length):
        self.char_to_index = char_to_index
        self.index_to_char = {index: char for char,
                              index in char_to_index.items()}
        self.index_to_char[char_to_index[SPECIAL_PAD]] = '@'
        self.max_input_length = max_input_length

    def tokenize(self, input_string):
        if len(input_string) > self.max_input_length:
            print(f"Warning: Input string is longer than max input length of {self.max_input_length}: {input_string}")
            input_string = input_string[:self.max_input_length]
        input_indices = [self.char_to_index.get(
            char, self.char_to_index[SPECIAL_PAD]) for char in input_string]

        # Pad the input to max_input_length using zeros
        total_pad = self.max_input_length - len(input_indices)
        # num_to_pad_left = random.randint(0, total_pad)
        num_to_pad_left = 0
        num_to_pad_right = total_pad - num_to_pad_left

        input_indices = [self.char_to_index[SPECIAL_PAD]]*num_to_pad_left + \
            input_indices + [self.char_to_index[SPECIAL_PAD]]*num_to_pad_right

        if len(input_indices) != self.max_input_length:
            raise ValueError(
                f"Input string is unexpected length: {input_string}: {len(input_indices)}")

        return np.array(input_indices)

    def indices_to_string(self, indices):
        return "".join([self.index_to_char[index] for index in indices])

    def tokenize_and_mask_input(self, input_string):
        input_tensor = self.tokenize(input_string)
        return input_tensor


class MaskedCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, char_to_index, **kwargs):
        super().__init__(**kwargs)
        self.char_to_index = char_to_index

    def call(self, y_true, y_pred):
        padding_mask = K.cast(
            K.equal(K.argmax(y_true, axis=-1), self.char_to_index[SPECIAL_PAD]), K.floatx())
        mask = 1.0 - 0.99 * padding_mask
        loss = keras.losses.categorical_crossentropy(y_true, y_pred)
        weighted_loss = loss * mask
        return K.sum(weighted_loss) / K.sum(mask)


def main(args):
    character_autoencoder_training()


def character_autoencoder_training():
    data_dir = Path(project_config['data_dir'])
    csv_path = data_dir / 'entities.csv'
    max_input_length = project_config['entities']['max_entity_length']
    batch_size = project_config['search_autoencoder']['batch_size']
    character_embedding_dim = project_config['search_autoencoder']['character_embedding_dim']

    alphabet = get_alphabet(csv_path)

    print_project_config(project_config)

    alphabet = [SPECIAL_PAD] + alphabet

    char_to_index = {char: index for index, char in enumerate(alphabet)}
    index_to_char = {index: char for index, char in enumerate(alphabet)}

    fresh_model = kermit_autoencoder(input_length=max_input_length,
                                     alphabet_size=len(alphabet),
                                     character_embedding_dim=character_embedding_dim)

    model_path = data_dir / "models" / "imdb_autoencoder.keras"
    loaded_model = try_load_model(model_path)

    if loaded_model is None:
        print("\n> no model found, using new model\n")
        model = fresh_model
    elif not are_models_same(fresh_model, loaded_model):
        print("\n> loaded model is out of date, using new model\n")
        model = fresh_model
    else:
        model = loaded_model

    print("\n> compiling model\n")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0002),
        loss=MaskedCategoricalCrossentropy(char_to_index=char_to_index),
        metrics=["accuracy"],
    )

    model.summary()

    token_processor = TokenProcessor(char_to_index=char_to_index, max_input_length=max_input_length)

    dataset = tf.data.Dataset.from_generator(lambda: autoencoder_batch_generator(csv_path, batch_size, token_processor),
                                             output_types=(
                                                 tf.int32, tf.float32),
                                             output_shapes=((batch_size, max_input_length, ),
                                                            (batch_size, max_input_length, len(char_to_index))))

    save_frequency = 200

    save_model_by_batch_callback = keras.callbacks.ModelCheckpoint(
        filepath=str(model_path),
        save_best_only=False,
        monitor="loss",
        save_freq=save_frequency,
        verbose=1
    )

    save_model_callback = keras.callbacks.ModelCheckpoint(
        filepath=str(model_path),
        save_best_only=False,
        monitor="loss",
        verbose=1
    )

    now_int = int(datetime.now().timestamp())
    folder_suggestion = suggest_folder_name()
    folder_suggestion = f"{now_int}_{folder_suggestion}"
    logs_dir = Path("logs") / folder_suggestion
    logs_dir.mkdir(parents=True, exist_ok=True)

    reconstruct_callback = ReconstructCallback(
        csv_path, token_processor, num_samples=10, frequency=save_frequency)

    tensorboard_callback = TensorBoardBatchLogger(
        log_dir=str(logs_dir), update_freq=5)

    commit_changes_to_git_callback = CommitChangesToGitCallback(logs_dir)

    learning_rate_callback = ScheduledLearningRateCallback(
        schedule=[
            (0, 0.0001),
            (1, 0.00008),
            (2, 0.00005),
            (3, 0.00002),
            (4, 0.00001),
            (5, 0.00008),
            (6, 0.00005),
            (7, 0.00002),
            (8, 0.00001)
        ]
    )

    model.fit(dataset, epochs=3, callbacks=[
        learning_rate_callback,
        reconstruct_callback,
        save_model_callback,
        commit_changes_to_git_callback,
        save_model_by_batch_callback,
        tensorboard_callback
    ])
    

def gelu_activation(x):
    return tf.keras.activations.gelu(x, approximate=True)


class StandardDeviationLayer(Layer):
    def call(self, inputs):
        return tf.math.reduce_std(inputs, axis=1, keepdims=True)


class CoefficientOfVariationLayer(Layer):
    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=1, keepdims=True)
        std_dev = tf.math.reduce_std(inputs, axis=1, keepdims=True)
        cv = std_dev / mean
        return tf.squeeze(cv, axis=1)


class EntropyLayer(Layer):
    def call(self, inputs):
        std_dev = tf.math.reduce_std(inputs, axis=1, keepdims=True)
        entropy = 0.5 * \
            tf.math.log(2 * tf.constant(np.pi) *
                        tf.constant(np.e) * tf.square(std_dev))

        entropy = tf.squeeze(entropy, axis=1)
        return entropy


class SinActivation(Layer):
    def call(self, inputs):
        return tf.math.sin(inputs)


class Sigmoid(Layer):
    def call(self, inputs):
        return tf.math.sigmoid(inputs)


class ReduceMeanLayer(Layer):
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=1, keepdims=True)


class SirenLayer(Layer):
    def __init__(self, units, name, is_first=False, omega=.5, activation=tf.math.sin, dtype=tf.float32):
        super(SirenLayer, self).__init__(name=f"{name}_siren")
        self.units = units
        self.layer_dtype = dtype
        self.is_first = is_first
        self.omega = omega
        self.activation = activation

    def build(self, input_shape):
        # Create a dense layer with the given number of units and sine activation
        self.layer = tf.keras.layers.Dense(
            units=self.units, activation=self.activation, dtype=self.layer_dtype)

        fan_in = input_shape[-1]

        # Initialize the weights of the layer according to the SIREN scheme
        if self.is_first:
            self.layer.kernel_initializer = tf.initializers.RandomUniform(minval=-tf.sqrt(1 / fan_in),
                                                                          maxval=tf.sqrt(1 / fan_in))
        else:
            self.layer.kernel_initializer = tf.initializers.RandomUniform(minval=-tf.sqrt(6 / fan_in) / self.omega,
                                                                          maxval=tf.sqrt(6 / fan_in) / self.omega)

    def call(self, inputs):
        # Call the dense layer with the given inputs
        return self.layer(inputs)

    def get_config(self):
        # Define the configuration of the layer
        config = super(SirenLayer, self).get_config()
        config.update({'units': self.units, 'dtype': self.layer_dtype})
        return config


class ExpandDims(Layer):
    def call(self, inputs):
        return tf.expand_dims(inputs, axis=1)


class Stack(Layer):
    """inputs are intended to be a list of tensors of the same shape"""

    def call(self, inputs):
        return tf.stack(inputs, axis=1)



def transformer_encoder_block(input, num_heads, ff_dim, dropout_rate=0.1):
    # Self-attention
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(input, input)
    attention_output = layers.Dropout(dropout_rate)(attention_output)
    attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + input)
    
    # Feed-forward network
    ff_output = layers.Dense(ff_dim, activation='gelu')(attention_output)
    ff_output = layers.Dense(input.shape[-1], activation='gelu')(ff_output)
    ff_output = layers.Dropout(dropout_rate)(ff_output)
    ff_output = layers.LayerNormalization(epsilon=1e-6)(ff_output + attention_output)
    
    return ff_output

def transformer_decoder_block(input, encoder_output, num_heads, ff_dim, dropout_rate=0.1):
    # Self-attention
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(input, input)
    attention_output = layers.Dropout(dropout_rate)(attention_output)
    attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + input)
    
    # Feed-forward network
    ff_output = layers.Dense(ff_dim, activation='gelu')(attention_output)
    ff_output = layers.Dense(input.shape[-1], activation='gelu')(ff_output)
    ff_output = layers.Dropout(dropout_rate)(ff_output)
    ff_output = layers.LayerNormalization(epsilon=1e-6)(ff_output + attention_output)
    
    return ff_output

def attention_pooling(inputs, num_heads, dropout_rate=0.1):
    # Self-attention to get attention scores
    attention_scores = layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1], dropout=dropout_rate)(inputs, inputs)
    attention_scores = layers.Softmax(axis=1)(attention_scores)  # Normalize scores across the sequence
    
    # Apply attention scores as a mask and multiply
    masked_inputs = layers.Multiply()([inputs, attention_scores])
    
    # reduce sum over the masked input sequence
    pooled_output = ReduceSum(axis=1)(masked_inputs)
    
    return pooled_output

def kermit_autoencoder(input_length, alphabet_size, character_embedding_dim, dropout_rate=0.001):
    num_encoder_blocks = 4
    num_decoder_blocks = 4
    num_heads = 1

    ff_dim = character_embedding_dim
    bottleneck_dim = character_embedding_dim

    inputs = Input(shape=(input_length,), dtype=tf.int32)
    embedding = Embedding(input_dim=alphabet_size, output_dim=character_embedding_dim)(inputs)
    embedding = layers.Dropout(dropout_rate)(embedding)

    reduced_to_half = Conv1D(character_embedding_dim, 3, strides=2, padding='same', activation='gelu')(embedding)
    reduced_to_quarter = Conv1D(character_embedding_dim, 3, strides=2, padding='same', activation='gelu')(reduced_to_half)

    reduced_to_quarter = LayerNormalization()(reduced_to_quarter)

    # Positional encoding
    positional_encoding = sinusoidal_encoding(input_length//4, character_embedding_dim)
    x = reduced_to_quarter + positional_encoding

    # Encoder blocks
    for _ in range(num_encoder_blocks):
        x = transformer_encoder_block(x, num_heads, ff_dim, dropout_rate)

    encoder_output = x
    encoder_output += positional_encoding
    # Bottleneck with combined attention pooling
    bottleneck = attention_pooling(encoder_output, num_heads, dropout_rate)
    bottleneck = layers.Dense(bottleneck_dim, activation='linear')(bottleneck)
    bottleneck = layers.LayerNormalization()(bottleneck)

    # Expand the bottleneck back to the sequence length for the decoder
    repeated_bottleneck = layers.RepeatVector(input_length//4)(bottleneck)
    
    x = repeated_bottleneck + positional_encoding
    x = Dense(character_embedding_dim, activation='gelu')(x)

    # Decoder blocks
    for _ in range(num_decoder_blocks):
        x = transformer_decoder_block(x, encoder_output, num_heads, ff_dim, dropout_rate)

    back_to_half = Conv1DTranspose(character_embedding_dim, 3, strides=2, padding='same', activation='gelu')(x)
    back_to_full = Conv1DTranspose(character_embedding_dim, 3, strides=2, padding='same', activation='gelu')(back_to_half)

    # Final output layer
    outputs = Dense(alphabet_size, activation='softmax')(back_to_full)

    model = Model(inputs=inputs, outputs=outputs, name="transformer_autoencoder_with_combined_attention_pooling")
    return model



class RecursiveTextBatchGenerator:
    def __init__(self, root_dir: str, batch_size: int, max_input_length: int, char_to_index: dict):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.max_input_length = max_input_length
        self.char_to_index = char_to_index
        self.file_list = self._get_all_txt_files()
        self.current_file_index = 0
        self.current_file = None
        self.current_file_content = ""
        self.current_position = 0

    def _get_all_txt_files(self) -> List[str]:
        txt_files = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".txt"):
                    txt_files.append(os.path.join(root, file))
        return txt_files

    def _load_next_file(self):
        if self.current_file_index < len(self.file_list):
            self.current_file = self.file_list[self.current_file_index]
            with open(self.current_file, 'r', encoding='utf-8') as f:
                self.current_file_content = f.read()
            self.current_position = 0
            self.current_file_index += 1
        else:
            self.current_file_content = ""
            self.current_position = 0

    def _get_next_sequence(self) -> str:
        if self.current_position >= len(self.current_file_content):
            self._load_next_file()
            if not self.current_file_content:
                return ""

        end_position = min(self.current_position + self.max_input_length, len(self.current_file_content))
        sequence = self.current_file_content[self.current_position:end_position]
        self.current_position = end_position
        return sequence

    def _tokenize(self, sequence: str) -> np.ndarray:
        tokenized = [self.char_to_index.get(char, self.char_to_index[SPECIAL_PAD]) for char in sequence]
        tokenized = tokenized[:self.max_input_length]  # Truncate if necessary
        tokenized += [self.char_to_index[SPECIAL_PAD]] * (self.max_input_length - len(tokenized))  # Pad if necessary
        return np.array(tokenized)

    def generate(self) -> tf.data.Dataset:
        def gen_func():
            while True:
                x_batch = np.zeros((self.batch_size, self.max_input_length), dtype=np.int32)
                y_batch = np.zeros((self.batch_size, self.max_input_length, len(self.char_to_index)), dtype=np.float32)

                for i in range(self.batch_size):
                    sequence = self._get_next_sequence()
                    if not sequence:
                        # If we've reached the end of all files, start over
                        self.current_file_index = 0
                        self._load_next_file()
                        sequence = self._get_next_sequence()

                    tokenized = self._tokenize(sequence)
                    x_batch[i] = tokenized
                    for j, token_index in enumerate(tokenized):
                        y_batch[i, j, token_index] = 1

                yield x_batch, y_batch

        return tf.data.Dataset.from_generator(
            gen_func,
            output_types=(tf.int32, tf.float32),
            output_shapes=((self.batch_size, self.max_input_length),
                           (self.batch_size, self.max_input_length, len(self.char_to_index)))
        )


def autoencoder_batch_generator(csv_path, batch_size, token_processor: TokenProcessor):
    while True:
        # Set up batches
        x_batch = np.zeros(
            (batch_size, token_processor.max_input_length), dtype=np.int32)
        y_batch = np.zeros((batch_size, token_processor.max_input_length, len(
            token_processor.char_to_index)), dtype=np.float32)

        # Open the CSV file
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row

            # Initialize batch index
            batch_index = 0

            print("Entity Batch Generator: Fetching entity names from CSV file...")
            for row in reader:
                entity_name = row[0].strip()

                if len(entity_name) > token_processor.max_input_length or len(entity_name) == 0:
                    continue

                try:
                    input_string_token_indices = token_processor.tokenize(entity_name)

                    x_batch[batch_index] = input_string_token_indices
                    for idx, token_index in enumerate(input_string_token_indices):
                        y_batch[batch_index, idx, token_index] = 1

                    batch_index += 1

                    if batch_index == batch_size:
                        batch_index = 0
                        yield x_batch, y_batch
                        x_batch = np.zeros(
                            (batch_size, token_processor.max_input_length), dtype=np.int32)
                        y_batch = np.zeros((batch_size, token_processor.max_input_length, len(
                            token_processor.char_to_index)), dtype=np.float32)
                except Exception as e:
                    traceback.print_exc()
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    print(f"{exc_type} {exc_obj} {exc_tb.tb_lineno}")
                    print(f"\nFailed to process entity name: {entity_name}\n")


def masked_categorical_crossentropy(char_to_index, y_true, y_pred):
    # Identify padding tokens (which should have reduced weight)
    padding_mask = K.cast(
        K.equal(K.argmax(y_true, axis=-1), char_to_index[SPECIAL_PAD]), K.floatx())

    mask = 1.0 - 0.99 * padding_mask

    # Calculate categorical crossentropy
    loss = keras.losses.categorical_crossentropy(y_true, y_pred)

    # Apply the weighted mask to the loss
    weighted_loss = loss * mask

    # Calculate mean loss, considering only the non-zero weights in the mask
    return K.sum(weighted_loss) / K.sum(mask)


class CommitChangesToGitCallback(keras.callbacks.Callback):
    def __init__(self, logs_dir):
        super().__init__()
        self.has_run = False
        self.logs_dir = logs_dir

    def on_batch_end(self, batch, logs=None):
        if not self.has_run:
            try:
                commit_changes_to_git(self.logs_dir)
            except Exception as e:
                print(f"Error committing changes to git: {e}")
                traceback.print_exc()
            finally:
                self.has_run = True


class TensorBoardBatchLogger(keras.callbacks.Callback):
    def __init__(self, log_dir, update_freq=100):
        super().__init__()
        self.log_dir = log_dir
        self.update_freq = update_freq
        self.writer = tf.summary.create_file_writer(log_dir)
        self.step = 0

    def on_batch_end(self, batch, logs=None):
        if self.step % self.update_freq == 0:
            with self.writer.as_default():
                for name, value in logs.items():
                    tf.summary.scalar(name, value, step=self.step)
                self.writer.flush()
        self.step += 1


class ReconstructCallback(keras.callbacks.Callback):
    def __init__(self, csv_path, token_processor: TokenProcessor, num_samples, frequency):
        super().__init__()
        self.csv_path = csv_path
        self.token_processor = token_processor
        self.num_samples = num_samples
        self.frequency = frequency
        self.all_entities = self._load_entities_from_csv()

    def _load_entities_from_csv(self):
        entities = []
        with open(self.csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                entity_name = row[0].strip()
                if len(entity_name) < self.token_processor.max_input_length:
                    entities.append(entity_name)
        return entities

    def on_batch_end(self, epoch, logs=None):
        if epoch % self.frequency != 0:
            return

        # Select a random sample of entity names
        selected_rows = random.sample(self.all_entities, self.num_samples)

        print(f"\nReconstruction examples at the end of epoch {epoch}:")

        for original in selected_rows:
            tokenized = self.token_processor.tokenize(original)
            # Reshape to (1, max_input_length)
            tokenized = np.expand_dims(tokenized, axis=0)
            prediction = self.model.predict(tokenized, verbose=0)
            reconstructed = self.token_processor.indices_to_string(
                np.argmax(prediction, axis=-1)[0])

            original_restored = self.token_processor.indices_to_string(
                tokenized[0])

            print(f"Original      | {original_restored}")
            print(f"Reconstructed | {reconstructed}\n")

        if epoch % 100 == 0:
            self.model.save(f"imdb_autoencoder_epoch_{epoch}.h5")
            print(f"Saved model at epoch {epoch}")


def are_models_same(model1, model2):
    """Compare two models internal shapes"""
    model1_shapes = [layer.shape for layer in model1.get_weights()]
    model2_shapes = [layer.shape for layer in model2.get_weights()]

    return model1_shapes == model2_shapes


def try_load_model(model_path: Path):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        custom_objects = {
            "DepthwiseConv1D": DepthwiseConv1D,
            "masked_categorical_crossentropy": masked_categorical_crossentropy,
        }
        return keras.models.load_model(model_path, custom_objects=custom_objects)
    except Exception as e:
        # print the exception
        traceback.print_exc()
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(f"{exc_type} {exc_obj} {exc_tb.tb_lineno}")

        print(f"Failed to load model from {model_path}")
        return None


class ScheduledLearningRateCallback(keras.callbacks.Callback):
    """Set learning rate according to the provided schedule."""

    def __init__(self, schedule):
        """
        schedule: List of tuples (epoch, learning_rate)
        """
        super().__init__()
        self.schedule = sorted(schedule, key=lambda x: x[0])

    def on_epoch_begin(self, epoch, logs=None):
        for scheduled_epoch, new_learning_rate in reversed(self.schedule):
            if epoch >= scheduled_epoch:
                if hasattr(self.model.optimizer, "learning_rate"):
                    self.model.optimizer.learning_rate.assign(
                        new_learning_rate)
                elif hasattr(self.model.optimizer, "lr"):
                    self.model.optimizer.lr.assign(new_learning_rate)
                print(f"Setting learning rate to {new_learning_rate}")
                break


def get_corpus_stats(corpus_stats_filename):
    """Load the corpus stats from the json file"""
    with open(corpus_stats_filename, "r", encoding="utf-8") as file:
        corpus_stats = json.load(file)
    return corpus_stats["alphabet"], corpus_stats["character_counts"]


def get_alphabet(csv_path):
    # Initialize a set to collect all unique characters
    alphabet_set = set()

    # Open the CSV file and read it
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row

        # Iterate over each row to collect characters
        for row in reader:
            entity_name = row[0].strip()
            if entity_name:  # Ensure that the entity name is not empty
                alphabet_set.update(entity_name)

    # Convert the set to a sorted list of characters
    alphabet = sorted(alphabet_set)

    return alphabet


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    main(args)
