from datetime import datetime
from functools import partial
import json
from pprint import pprint
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


def character_autoencoder_training():
    data_dir = Path(project_config['data_dir'])
    jsonl_files = [data_dir / jsonl_file for jsonl_file in project_config['dataset']['jsonl_files']]
    pprint(project_config)
    max_input_length = project_config['entities']['max_entity_length']
    max_output_length = project_config['siren']['max_output_length']
    batch_size = project_config['siren']['batch_size']
    epochs = project_config['siren']['epochs']

    # Prepare alphabet and character mapping from multiple JSONL files
    alphabet = get_alphabet(jsonl_files)
    alphabet = [SPECIAL_PAD] + alphabet

    char_to_index = {char: index for index, char in enumerate(alphabet)}

    model = kermit_siren(input_length=max_input_length,
                               alphabet_size=len(alphabet),
                               config=project_config['siren'])

    model_path = data_dir / "models" / "imdb_autoencoder.keras"

    print("\n> compiling model\n")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0002),
        loss=MaskedCategoricalCrossentropy(char_to_index=char_to_index),
        metrics=["accuracy"],
    )

    model.summary()

    input_tokenizer = Tokenizer(char_to_index=char_to_index, max_input_length=max_input_length)
    output_tokenizer = Tokenizer(char_to_index=char_to_index, max_input_length=max_output_length)

    # Use the new batch generator for multiple JSONL files
    dataset = tf.data.Dataset.from_generator(
        lambda: autoencoder_batch_generator(jsonl_files, batch_size, input_tokenizer, output_tokenizer),
        output_types=(tf.int32, tf.float32),
        output_shapes=((batch_size, max_input_length,), (batch_size, max_output_length, len(char_to_index)))
    )

    # Define callbacks
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
        jsonl_files, input_tokenizer, num_samples=10, frequency=save_frequency
    )

    tensorboard_callback = TensorBoardBatchLogger(
        log_dir=str(logs_dir), update_freq=5)

    commit_changes_to_git_callback = CommitChangesToGitCallback(logs_dir)

    learning_rate_callback = ScheduledLearningRateCallback(
        schedule=[
            (0, 0.0002),
            (1, 0.0001),
            (2, 0.00005),
            # (3, 0.00002),
            # (4, 0.00001),
            # (5, 0.00008),
            # (6, 0.00005),
            # (7, 0.00002),
            # (8, 0.00001)
        ]
    )

    # Train the model
    model.fit(dataset, epochs=epochs, callbacks=[
        learning_rate_callback,
        reconstruct_callback,
        # save_model_callback,
        # save_model_by_batch_callback,
        commit_changes_to_git_callback,
        tensorboard_callback
    ])


class Tokenizer:
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
            units=self.units, activation=None, dtype=self.layer_dtype)  # Removed activation to apply it after adding dynamic bias

        fan_in = input_shape[0][-1]  # Input shape is a tuple of (position_encoding, bottleneck), so use the first

        # Initialize the weights of the layer according to the SIREN scheme
        if self.is_first:
            self.layer.kernel_initializer = tf.initializers.RandomUniform(minval=-tf.sqrt(1 / fan_in),
                                                                          maxval=tf.sqrt(1 / fan_in))
        else:
            self.layer.kernel_initializer = tf.initializers.RandomUniform(minval=-tf.sqrt(6 / fan_in) / self.omega,
                                                                          maxval=tf.sqrt(6 / fan_in) / self.omega)

        # Dense layer to generate dynamic biases from the bottleneck
        self.bias_dense = tf.keras.layers.Dense(self.units, dtype=self.layer_dtype, activation=tf.math.sin)
        self.frequency_dense = tf.keras.layers.Dense(self.units, dtype=self.layer_dtype, activation=tf.math.sin)

    def call(self, inputs):
        # Split inputs into positional encoding and bottleneck
        position_encoding, bottleneck = inputs

        # Pass the bottleneck through the bias-generating dense layer
        dynamic_bias = self.bias_dense(bottleneck)

        # Generate dynamic frequency scaling factors
        dynamic_frequency = self.frequency_dense(bottleneck)  # Shape: [batch_size, units]

        # Ensure dynamic bias has the correct shape by expanding dimensions
        dynamic_bias = tf.expand_dims(dynamic_bias, axis=1)  # Shape: [batch_size, 1, units]
        dynamic_frequency = tf.expand_dims(dynamic_frequency, axis=1)  # Shape: [batch_size, 1, units]

        # Repeat to match positional encoding length
        dynamic_bias = tf.repeat(dynamic_bias, repeats=tf.shape(position_encoding)[1], axis=1)  # Shape: [batch_size, output_length, units]
        dynamic_frequency = tf.repeat(dynamic_frequency, repeats=tf.shape(position_encoding)[1], axis=1)  # Shape: [batch_size, output_length, units]

        # Compute the weighted sum of inputs
        weighted_input = self.layer(position_encoding)

        # Adjust the weighted input with dynamic frequency
        biased_output = dynamic_frequency * weighted_input + dynamic_bias

        # Apply sine activation function
        output = self.activation(self.omega * biased_output)

        return output


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


def attention_pooling(inputs, num_heads, positional_encoding, dropout_rate=0.1):
    # Self-attention to get attention scores

    positional_encoded = positional_encoding + inputs
    attention_scores = layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1], dropout=dropout_rate)(positional_encoded, inputs)
    attention_scores = layers.Softmax(axis=1)(attention_scores)  # Normalize scores across the sequence
    
    # Apply attention scores as a mask and multiply
    masked_inputs = layers.Multiply()([inputs, attention_scores])
    
    # reduce sum over the masked input sequence
    pooled_output = ReduceSum(axis=1)(masked_inputs)
    
    return pooled_output


class ExpandAndTileLayer(Layer):
    def __init__(self, output_length, **kwargs):
        super(ExpandAndTileLayer, self).__init__(**kwargs)
        self.output_length = output_length

    def call(self, bottleneck):
        normalized_positions = tf.range(0, self.output_length, dtype=tf.float32) / self.output_length
        expanded_positions = tf.tile(tf.expand_dims(normalized_positions, axis=0), [tf.shape(bottleneck)[0], 1])
        expanded_positions = tf.expand_dims(expanded_positions, axis=-1)
        return expanded_positions

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_length, 1)


class RepeatBottleneckLayer(Layer):
    def __init__(self, output_length, **kwargs):
        super(RepeatBottleneckLayer, self).__init__(**kwargs)
        self.output_length = output_length

    def call(self, bottleneck):
        expanded_bottleneck = tf.repeat(tf.expand_dims(bottleneck, axis=1), self.output_length, axis=1)
        return expanded_bottleneck

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_length, input_shape[-1])

def kermit_siren(input_length, alphabet_size, config: dict):
    num_heads = 1

    character_embedding_dim = config['character_embedding_dim']
    bottleneck_dim = config['bottleneck_dim']
    output_length = config['max_output_length']
    siren_width = config['siren_width']
    num_encoder_blocks = config['num_encoder_blocks']
    dropout_rate = 0.01

    ff_dim = character_embedding_dim

    inputs = Input(shape=(input_length,), dtype=tf.int32)
    embedding_layer = Embedding(input_dim=alphabet_size, output_dim=character_embedding_dim)
    embedding = embedding_layer(inputs)
    dropped_out_embedding = Dropout(dropout_rate)(embedding)

    reduced_to_half = Conv1D(character_embedding_dim, 3, strides=2, padding='same', activation='gelu')(dropped_out_embedding)
    reduced_to_quarter = Conv1D(character_embedding_dim, 3, strides=2, padding='same', activation='gelu')(reduced_to_half)

    reduced_to_quarter = LayerNormalization()(reduced_to_quarter)

    # Positional encoding
    positional_encoding = sinusoidal_encoding(input_length // 4, character_embedding_dim)
    x = reduced_to_quarter + positional_encoding

    # Encoder blocks
    for _ in range(num_encoder_blocks):
        x = transformer_encoder_block(x, num_heads, ff_dim, dropout_rate)

    encoder_output = x
    # Bottleneck with combined attention pooling
    bottleneck = attention_pooling(encoder_output, num_heads, positional_encoding, dropout_rate)

    # Expansion of the bottleneck
    bottleneck = Dense(bottleneck_dim * 4, activation=tf.math.sin)(bottleneck)  # Use GELU or another activation to add non-linearity

    siren_positional_encoding = sinusoidal_encoding(output_length, character_embedding_dim)

    # Initialize SIREN layer input
    siren_output = siren_positional_encoding

    # Pass through SIREN layers
    for i in range(config['num_siren_layers']):
        siren_output = SirenLayer(siren_width, name=f"siren_{i}", is_first=(i == 0))([siren_output, bottleneck])

    siren_output = SirenLayer(character_embedding_dim, name="siren_output", is_first=False)([siren_output, bottleneck])

    # Apply the embedding layer to convert the output to logits over the vocabulary
    x = Lambda(lambda x: tf.matmul(x, embedding_layer.embeddings, transpose_b=True), output_shape=(output_length, alphabet_size))(siren_output)
    x = Softmax()(x)

    model = Model(inputs=inputs, outputs=x)
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


def autoencoder_batch_generator(jsonl_files, batch_size, input_tokenizer: Tokenizer, output_tokenizer: Tokenizer):
    """
    Efficient batch generator for autoencoder training using multiple JSONL files.
    """
    # Load all JSONL data into memory
    all_data = []
    dataset_sizes = []

    # Load data from each JSONL file and store in memory
    for jsonl_path in jsonl_files:
        with open(jsonl_path, 'r', encoding='utf-8') as jsonl_file:
            data = [json.loads(line) for line in jsonl_file]
            all_data.append(data)
            dataset_sizes.append(len(data))
    
    # Calculate sampling probabilities based on dataset sizes
    total_size = sum(dataset_sizes)
    sampling_probs = [size / total_size for size in dataset_sizes]

    # Generate batches
    data_indices = [0] * len(jsonl_files)  # Track the current index for each dataset
    x_batch = np.zeros((batch_size, input_tokenizer.max_input_length), dtype=np.int32)
    y_batch = np.zeros((batch_size, output_tokenizer.max_input_length, len(output_tokenizer.char_to_index)), dtype=np.float32)

    batch_index = 0

    while True:
        # Randomly choose a dataset based on sampling probabilities
        dataset_choice = random.choices(range(len(jsonl_files)), weights=sampling_probs, k=1)[0]

        # Check if we've exhausted the chosen dataset
        if data_indices[dataset_choice] >= dataset_sizes[dataset_choice]:
            continue  # Skip if this dataset is exhausted

        # Fetch data point from chosen dataset
        data_point = all_data[dataset_choice][data_indices[dataset_choice]]
        data_indices[dataset_choice] += 1

        query = data_point.get('query', '').strip()
        result = data_point.get('result', '').strip()

        if not query or len(query) > input_tokenizer.max_input_length:
            continue

        try:
            # Tokenize the input query
            input_string_token_indices = input_tokenizer.tokenize(query)

            # Tokenize the output result
            output_token_indices = output_tokenizer.tokenize(result)[:output_tokenizer.max_input_length]

            # Update the batch
            x_batch[batch_index] = input_string_token_indices

            # One-hot encode the output batch
            for idx, token_index in enumerate(output_token_indices):
                y_batch[batch_index, idx, token_index] = 1

            batch_index += 1

            # Yield a full batch
            if batch_index == batch_size:
                batch_index = 0
                yield x_batch, y_batch
                # Reset for next batch
                x_batch = np.zeros((batch_size, input_tokenizer.max_input_length), dtype=np.int32)
                y_batch = np.zeros((batch_size, output_tokenizer.max_input_length, len(output_tokenizer.char_to_index)), dtype=np.float32)

        except Exception as e:
            traceback.print_exc()
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"{exc_type} {exc_obj} {exc_tb.tb_lineno}")
            print(f"\nFailed to process entity: {query}\n")
        
        # Check if all datasets are exhausted
        if all(index >= size for index, size in zip(data_indices, dataset_sizes)):
            break  # Exit loop if all datasets are exhausted


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
    def __init__(self, jsonl_files, input_tokenizer: Tokenizer, num_samples, frequency):
        super().__init__()
        self.jsonl_files = jsonl_files
        self.input_tokenizer = input_tokenizer
        self.num_samples = num_samples
        self.frequency = frequency
        self.all_entities = self._load_entities_from_jsonl()

    def _load_entities_from_jsonl(self):
        entities = []
        types = []
        for jsonl_path in self.jsonl_files:
            with open(jsonl_path, 'r', encoding='utf-8') as jsonl_file:
                for line in jsonl_file:
                    data = json.loads(line)
                    entity_name = data.get('query', '').strip()
                    entity_type = data.get('type', '').strip()
                    if len(entity_name) < self.input_tokenizer.max_input_length:
                        entities.append(entity_name)
                        types.append(entity_type)
        return list(zip(entities, types))

    def on_batch_end(self, epoch, logs=None):
        if epoch % self.frequency != 0:
            return

        # Select a random sample of entity names
        selected_rows = random.sample(self.all_entities, self.num_samples)

        print(f"\nReconstruction examples at the end of epoch {epoch}:")

        for original in selected_rows:
            tokenized = self.input_tokenizer.tokenize(original[0])
            # Reshape to (1, max_input_length)
            tokenized = np.expand_dims(tokenized, axis=0)
            prediction = self.model.predict(tokenized, verbose=0)
            reconstructed = self.input_tokenizer.indices_to_string(
                np.argmax(prediction, axis=-1)[0])

            original_restored = self.input_tokenizer.indices_to_string(
                tokenized[0])

            print(f"Type   | {original[1]}")
            print(f"Query  | {original_restored}")
            print(f"Answer | {reconstructed}\n")


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


def get_alphabet(jsonl_files, cache_file='alphabet_cache.json'):
    """
    Extracts all unique characters from multiple JSONL files to create an alphabet.
    Checks for a cached version first.
    """
    # Compute a hash of the JSONL files to use as a cache key
    hash_key = compute_files_hash(jsonl_files)
    
    # Check if the cache file exists and is up-to-date
    if Path(cache_file).exists():
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
            if cache_data.get('hash_key') == hash_key:
                print("Loading alphabet from cache.")
                return cache_data['alphabet']

    # Initialize a set to collect all unique characters
    alphabet_set = set()

    # Process each JSONL file to compute the alphabet
    for jsonl_path in jsonl_files:
        with open(jsonl_path, 'r', encoding='utf-8') as jsonl_file:
            for line in jsonl_file:
                data = json.loads(line)
                query = data.get('query', '').strip()
                result = data.get('result', '').strip()

                # Ensure that the strings are not empty and add their characters to the set
                if query:
                    alphabet_set.update(query)
                if result:
                    alphabet_set.update(result)

    # Convert the set to a sorted list of characters
    alphabet = sorted(alphabet_set)

    # Cache the computed alphabet along with the hash key
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump({'hash_key': hash_key, 'alphabet': alphabet}, f)

    return alphabet


def compute_files_hash(files):
    """
    Computes a hash of the contents of the given files to use as a cache key.
    """
    hasher = hashlib.sha256()
    for file_path in files:
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
    return hasher.hexdigest()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    main(args)
