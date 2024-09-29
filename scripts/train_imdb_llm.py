from datetime import datetime
from functools import partial
import json
import logging
from math import sqrt
import math
from pprint import pprint
import sqlite3
import sys
import time
import traceback
from typing import List, Tuple
# import opencv
import re
from tqdm import tqdm
from pathlib import Path
import numpy as np
from collections import defaultdict
import tensorflow as tf
from collections import deque
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention, LayerNormalization, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, UpSampling1D, Add, Reshape, BatchNormalization, Layer, Conv1DTranspose, DepthwiseConv1D, Conv1DTranspose, GlobalAveragePooling1D, Concatenate, Permute, Multiply, Add, Dense, Flatten, GlobalMaxPooling1D, Layer, Softmax, Lambda, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
Callback = keras.callbacks.Callback
from config import project_config
from datasets import load_dataset, concatenate_datasets
from scripts.attention_model import generative_matrix_attention
from scripts.utils import print_project_config
# from torch.utils.data import Dataset, DataLoader
import os
import json
import hashlib
import random
from typing import Set, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from track_changes import commit_changes_to_git, suggest_folder_name
from copy import deepcopy

tf_version = tf.__version__

SPECIAL_PAD = '\u200C'
SPECIAL_START = '\u200D'
SPECIAL_END = '\u200E'
SPECIAL_SEP = '\u200F'
import itertools

logging.basicConfig(level=logging.DEBUG)

def main():
    configs = generate_param_combinations(project_config)
    for idx, config in enumerate(configs):
        print(f"\nRunning configuration {idx + 1}/{len(configs)}:")
        name_prefix = f"{config['llm']['input_length']}_{config['llm']['num_heads']}"

        character_llm_training(config, name_prefix)


def generate_param_combinations(config):
    sweep_params = {}
    fixed_params = {}

    for key, value in config['llm'].items():
        if isinstance(value, list):
            sweep_params[key] = value
        else:
            fixed_params[key] = value

    keys, values = zip(*sweep_params.items()) if sweep_params else ([], [])
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)] if keys else [{}]

    configs = []
    for combo in combinations:
        new_config = deepcopy(config)
        new_config['llm'].update(fixed_params)
        new_config['llm'].update(combo)
        configs.append(new_config)

    return configs

def character_llm_training(config, name_prefix):
    # Initialize paths and configuration
    data_dir = Path(config['data_dir'])
    input_length = config['llm']['input_length']
    batch_size = config['llm']['batch_size']
    character_embedding_dim = config['llm']['character_embedding_dim']
    num_epochs = config['llm']['epochs']
    num_heads = config['llm']['num_heads']
    jsonl_files = [data_dir / jsonl_file for jsonl_file in config['dataset']['jsonl_files']]

    print_project_config(config)

    tokenizer = Tokenizer.from_files(jsonl_files, input_length)
    total_samples = calculate_steps_per_epoch(jsonl_files, tokenizer)
    steps_per_epoch = total_samples // batch_size

    # Define the checkpoint path (modify to include parameters if needed)
    checkpoint_dir = data_dir / "checkpoints" / f"input{input_length}_embed{character_embedding_dim}_batch{batch_size}_epochs{num_epochs}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "kermit_language_model.keras"

    # Load or create the model
    model = create_new_model(tokenizer, input_length, character_embedding_dim, num_heads)

    shared_buffer = deque(maxlen=1000)  # Adjust the size based on memory constraints
    dataset = autoencoder_batch_generator(jsonl_files, batch_size, tokenizer, shared_buffer)
    callbacks = get_callbacks(tokenizer, name_prefix, input_length, shared_buffer, checkpoint_path)

    # Compile and train model
    compile_and_train_model(model, tokenizer, dataset, steps_per_epoch, num_epochs, callbacks)


def load_or_create_model(checkpoint_path, tokenizer, input_length, character_embedding_dim, num_heads):
    """
    Attempt to load the model from the checkpoint. If it fails, create a new model.
    Logs the process and any errors encountered during model loading.
    """
    logging.info(f"Trying to load the model from {checkpoint_path}")
    custom_objects = {
        'WeightedLossWithPadding': WeightedLossWithPadding,
        'AlibiMultiHeadAttention': AlibiMultiHeadAttention,
    }

    try:
        if os.path.exists(checkpoint_path):
            logging.info(f"Checkpoint found at {checkpoint_path}. Loading the model.")
            model = load_model(checkpoint_path, custom_objects=custom_objects)
            logging.info("Model loaded successfully.")
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    except Exception as e:
        logging.error(f"Failed to load model from {checkpoint_path}. Error: {e}")
        logging.info("Creating a new model.")
        model = create_new_model(tokenizer, input_length, character_embedding_dim, num_heads)

    return model


def create_new_model(tokenizer, input_length, character_embedding_dim, num_heads):
    """
    Creates a new model instance and logs the creation process.
    """
    logging.info("Building a new model.")
    model = kermit_language_model(
        input_length=input_length,
        alphabet_size=tokenizer.alphabet_size,
        character_embedding_dim=character_embedding_dim,
        num_heads=num_heads,
    )
    logging.info("New model created.")
    return model

def get_callbacks(tokenizer, name_prefix, input_length, shared_buffer, model_filepath):
    reconstruct_callback = ReconstructCallback(
        tokenizer=tokenizer,
        shared_buffer=shared_buffer,
        max_input_length=input_length,
        num_samples=5,
        frequency=100,
    )

    logs_dir = create_logs_dir(name_prefix=name_prefix)
    tensorboard_callback = TensorBoardBatchLogger(log_dir=str(logs_dir), update_freq=5)
    commit_changes_to_git_callback = CommitChangesToGitCallback(logs_dir)

    # Define the checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        filepath=str(model_filepath),
        save_weights_only=False,
        save_freq=100,
    )

    return [
        commit_changes_to_git_callback,
        tensorboard_callback,
        reconstruct_callback,
        keras.callbacks.TensorBoard('./logs', update_freq=5),
        checkpoint_callback
    ]


def compile_and_train_model(model, tokenizer, dataset, steps_per_epoch, num_epochs, callbacks):
    initial_learning_rate = 0.0002
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=steps_per_epoch * num_epochs,
        alpha=0.0
    )

    # Get the padding token index from the tokenizer
    pad_token_index = tokenizer.char_to_index[SPECIAL_PAD]

    # Instantiate the custom loss function
    loss_fn = WeightedLossWithPadding(pad_token_index=pad_token_index, pad_weight=0.1)

    print("\n> Compiling model\n")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=loss_fn,
        metrics=['accuracy'],
    )

    model.summary()

    print("\n> Training model\n")
    model.fit(
        dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        callbacks=callbacks
    )

class WeightedLossWithPadding(tf.keras.losses.Loss):
    def __init__(self, pad_token_index, pad_weight=0.1, **kwargs):
        super().__init__(**kwargs)
        self.pad_token_index = pad_token_index
        self.pad_weight = pad_weight

    def call(self, y_true, y_pred):
        # Compute the standard sparse categorical cross-entropy loss
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

        # Create a mask for padding tokens
        mask = tf.cast(tf.not_equal(y_true, self.pad_token_index), tf.float32)

        # Apply the weights to the loss
        weights = mask + self.pad_weight * (1 - mask)
        weighted_loss = loss * weights

        # Return the mean loss over the batch
        return tf.reduce_mean(weighted_loss)

    def get_config(self):
        # Return a configuration dictionary that can be used for serialization
        config = super().get_config()
        config.update({
            'pad_token_index': self.pad_token_index,
            'pad_weight': self.pad_weight,
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Create an instance from the config
        return cls(**config)


def create_logs_dir(name_prefix):
    now_int = int(datetime.now().timestamp())
    folder_suggestion = f"{now_int}_{name_prefix}_{suggest_folder_name()}"
    logs_dir = Path("logs") / folder_suggestion
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


class Tokenizer:
    def __init__(self, char_to_index, index_to_char, max_input_length):
        self.char_to_index = char_to_index
        self.alphabet_size = len(char_to_index)
        self.index_to_char = index_to_char
        self.max_input_length = max_input_length

    @classmethod
    def from_files(cls, jsonl_files, max_input_length):
        alphabet = get_alphabet(jsonl_files) + [SPECIAL_PAD, SPECIAL_START, SPECIAL_END, SPECIAL_SEP]
        char_to_index = {char: index for index, char in enumerate(sorted(alphabet))}
        index_to_char = {index: char for char, index in char_to_index.items()}

        index_to_char[char_to_index[SPECIAL_PAD]] = '@'
        index_to_char[char_to_index[SPECIAL_START]] = '<|'
        index_to_char[char_to_index[SPECIAL_END]] = '|>'
        index_to_char[char_to_index[SPECIAL_SEP]] = '<|>'

        return cls(char_to_index, index_to_char, max_input_length)

    def tokenize(self, input_string, pad_to_max_length=True):
        input_indices = [self.char_to_index.get(char, self.char_to_index[SPECIAL_PAD]) for char in input_string]
        if pad_to_max_length:
            input_indices = input_indices[:self.max_input_length]
            pad_length = self.max_input_length - len(input_indices)
            if pad_length > 0:
                input_indices = [self.char_to_index[SPECIAL_PAD]] * pad_length + input_indices
        return input_indices  # Return as list for concatenation

    def untokenize(self, indices):
        return "".join(self.index_to_char.get(index, '') for index in indices)
    




query_type_to_query_fieldname = {
    'movie': 'title',
    'tvSeries': 'title',
    'person': 'name',
    'character': 'name',
}

def autoencoder_batch_generator(jsonl_files, batch_size, tokenizer: Tokenizer, shared_buffer: deque):
    """
    Batch generator that generates input-target pairs for language modeling.
    For each position in the full sequence, generates an input sequence consisting
    of the tokens up to that position, left-padded to max_input_length.
    The target sequence is the input sequence shifted one token to the right.
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

    # Initialize variables
    data_indices = [0] * len(jsonl_files)  # Track the current index for each dataset
    x_batch = np.zeros((batch_size, tokenizer.max_input_length), dtype=np.int32)
    y_batch = np.zeros((batch_size, tokenizer.max_input_length), dtype=np.int32)
    batch_index = 0

    while True:
        # Randomly choose a dataset based on sampling probabilities
        dataset_choice = random.choices(range(len(jsonl_files)), weights=sampling_probs, k=1)[0]

        # Check if we've exhausted the chosen dataset
        if data_indices[dataset_choice] >= dataset_sizes[dataset_choice]:
            data_indices[dataset_choice] = 0  # Reset index if exhausted

        # Fetch data point from chosen dataset
        data_point = all_data[dataset_choice][data_indices[dataset_choice]]
        shared_buffer.append(data_point)
        data_indices[dataset_choice] += 1

        query_type = data_point.get('type')

        if query_type not in ['movie', 'tvSeries', 'person', 'character']:
            logging.debug(f"Skipping data point due to unknown query type: {query_type}")
            continue

        query_fieldname = query_type_to_query_fieldname[query_type]
        query = data_point.get(query_fieldname, '').strip()

        result = json.dumps(data_point)

        # Build input and target sequences
        input_string = f"{SPECIAL_START}{query}{SPECIAL_SEP}{result}{SPECIAL_END}"

        # Build full sequence
        full_sequence = tokenizer.tokenize(input_string, pad_to_max_length=False)

        sequence_length = len(full_sequence)
        max_len = tokenizer.max_input_length

        if sequence_length < 2:
            continue  # Skip sequences that are too short

        for i in range(sequence_length - 1):
            # Input sequence is from the start to position i (inclusive)
            input_indices = full_sequence[:i+1]
            # Target sequence is from the start to position i+1 (inclusive)
            target_indices = full_sequence[:i+2]

            # Pad input_indices on the left to length max_len
            if len(input_indices) < max_len:
                pad_length = max_len - len(input_indices)
                input_indices = [tokenizer.char_to_index[SPECIAL_PAD]] * pad_length + input_indices
            else:
                input_indices = input_indices[-max_len:]

            # Pad target_indices on the left to length max_len
            if len(target_indices) < max_len:
                pad_length = max_len - len(target_indices)
                target_indices = [tokenizer.char_to_index[SPECIAL_PAD]] * pad_length + target_indices
            else:
                target_indices = target_indices[-max_len:]

            # Assign to batch arrays
            x_batch[batch_index, :] = input_indices
            y_batch[batch_index, :] = target_indices

            batch_index += 1

            if batch_index == batch_size:
                yield x_batch, y_batch
                batch_index = 0
                x_batch = np.zeros((batch_size, max_len), dtype=np.int32)
                y_batch = np.zeros((batch_size, max_len), dtype=np.int32)

        # Check if all datasets are exhausted
        if all(index >= size for index, size in zip(data_indices, dataset_sizes)):
            # Reset all indices and continue
            data_indices = [0] * len(jsonl_files)


def create_name_generator():
    name_counts = defaultdict(int)

    def generate_unique_name(name: str) -> str:
        name_counts[name] += 1
        count = name_counts[name]
        return f"{name}_{count}"

    return generate_unique_name

n = create_name_generator()

def sinusoidal_encoding(input_length, embedding_dim):
    positions = tf.range(input_length - 1, -1, -1, dtype=tf.float32)
    dimensions = tf.range(embedding_dim, dtype=tf.float32)
    
    angle_rates = 1 / tf.pow(10000.0, (2 * (dimensions // 2)) / tf.cast(embedding_dim, tf.float32))
    angle_rads = tf.expand_dims(positions, 1) * tf.expand_dims(angle_rates, 0)
    
    sines = tf.sin(angle_rads[:, 0::2])
    cosines = tf.cos(angle_rads[:, 1::2])
    
    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, ...]
    
    return pos_encoding


def transformer_encoder_block(input, num_heads, ff_dim, dropout_rate=0.1):
    """
    Transformer Encoder Block with ALiBi Multi-Head Attention.

    Args:
        input (tf.Tensor): Input tensor.
        num_heads (int): Number of attention heads.
        ff_dim (int): Dimensionality of the feed-forward layer.
        dropout_rate (float): Dropout rate.

    Returns:
        tf.Tensor: Output tensor after applying the encoder block.
    """
    # ALiBi Multi-Head Attention
    attention_output = AlibiMultiHeadAttention(num_heads=num_heads, key_dim=ff_dim, dropout=dropout_rate)(input, input, input)
    attention_output = layers.Dropout(dropout_rate)(attention_output)
    attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + input)
    
    # Feed-forward network
    ff_output = layers.Dense(ff_dim, activation='gelu')(attention_output)
    ff_output = layers.Dense(input.shape[-1], activation='gelu')(ff_output)
    ff_output = layers.Dropout(dropout_rate)(ff_output)
    ff_output = layers.LayerNormalization(epsilon=1e-6)(ff_output + attention_output)
    
    return ff_output

def transformer_decoder_block(input, num_heads, ff_dim, dropout_rate=0.1):
    model_dim = input.shape[-1]
    key_dim = model_dim //4 // num_heads  # Ensure key_dim divides model_dim evenly
    
    attention_output = AlibiMultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        model_dim=model_dim,
        dropout=dropout_rate
    )(input, input, input)
    
    attention_output = layers.Dropout(dropout_rate)(attention_output)
    attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + input)
    
    # Feed-forward network
    ff_output = layers.Dense(ff_dim, activation='gelu')(attention_output)
    ff_output = layers.Dense(model_dim, activation='gelu')(ff_output)
    ff_output = layers.Dropout(dropout_rate)(ff_output)
    ff_output = layers.LayerNormalization(epsilon=1e-6)(ff_output + attention_output)
    
    return ff_output


######################################################################################################################################
#-------------------------------------------------------------------------------------------------------------------------------------
#......................................................................................................................................

def kermit_language_model(input_length, alphabet_size, character_embedding_dim, num_heads, dropout_rate=0.001):
    num_decoder_blocks = 8
    
    inputs = Input(shape=(input_length,), dtype=tf.int32)
    embedding = Embedding(input_dim=alphabet_size, output_dim=character_embedding_dim)(inputs)
    
    reduced_to_half = layers.Conv1D(character_embedding_dim, 3, strides=2, padding='same', activation='gelu')(embedding)
    reduced_to_quarter = layers.Conv1D(character_embedding_dim, 3, strides=2, padding='same', activation='gelu')(reduced_to_half)
    reduced_to_quarter = layers.LayerNormalization()(reduced_to_quarter)
    
    x = reduced_to_quarter
    
    # Decoder blocks using ALiBi
    for _ in range(num_decoder_blocks):
        x = transformer_decoder_block(x, num_heads, ff_dim=character_embedding_dim, dropout_rate=dropout_rate)
    
    back_to_half = layers.Conv1DTranspose(character_embedding_dim, 3, strides=2, padding='same', activation='gelu')(x)
    back_to_full = layers.Conv1DTranspose(character_embedding_dim, 3, strides=2, padding='same', activation='gelu')(back_to_half)
    
    # Final output layer
    outputs = layers.Dense(alphabet_size, activation='softmax')(back_to_full)
    
    model = Model(inputs=inputs, outputs=outputs, name="transformer_autoencoder_with_alibi_attention")
    return model



#......................................................................................................................................
#-------------------------------------------------------------------------------------------------------------------------------------
######################################################################################################################################

import tensorflow as tf
from tensorflow.keras import layers
import math

class AlibiMultiHeadAttention(layers.Layer):
    def __init__(self, num_heads, key_dim, model_dim, dropout=0.0, **kwargs):
        super(AlibiMultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.model_dim = model_dim
        self.dropout = dropout
        self.dropout_layer = layers.Dropout(dropout)
        
        # Define the query, key, value dense layers
        self.query_dense = layers.Dense(self.num_heads * self.key_dim, use_bias=True)
        self.key_dense = layers.Dense(self.num_heads * self.key_dim, use_bias=True)
        self.value_dense = layers.Dense(self.num_heads * self.key_dim, use_bias=True)
        
        # Define the output dense layer to project back to model_dim
        self.output_dense = layers.Dense(self.model_dim)
        
        # ALiBi slopes
        self.alibi_slopes = self._get_slopes(self.num_heads)
    
    def _get_slopes(self, n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-2 ** -(math.log2(n) - 3))
            ratio = start
            return [start * ratio ** i for i in range(n)]
        if math.log2(n).is_integer():
            return tf.constant(get_slopes_power_of_2(n), dtype=tf.float32)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return tf.constant(
                get_slopes_power_of_2(closest_power_of_2) +
                get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:n - closest_power_of_2],
                dtype=tf.float32
            )
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.key_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, query, value, key, training=False):
        batch_size = tf.shape(query)[0]
        seq_len = tf.shape(query)[1]

        # Linear projections
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # Split into heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # Compute attention scores
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(self.key_dim, tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # Create causal mask
        i = tf.range(seq_len)[:, None]
        j = tf.range(seq_len)[None, :]
        causal_mask = tf.where(i >= j, 0.0, -1e9)
        causal_mask = tf.expand_dims(causal_mask, axis=0)  # [1, seq_len, seq_len]
        causal_mask = tf.expand_dims(causal_mask, axis=0)  # [1, 1, seq_len, seq_len]
        causal_mask = tf.tile(causal_mask, [batch_size, self.num_heads, 1, 1])

        # Compute relative positions
        relative_positions = i - j  # Shape: [seq_len, seq_len]
        relative_positions = tf.expand_dims(relative_positions, axis=0)  # [1, seq_len, seq_len]

        # Reshape alibi slopes
        alibi_slopes = tf.reshape(self.alibi_slopes, [self.num_heads, 1, 1])  # [num_heads, 1, 1]

        # Compute ALiBi biases
        alibi_bias = alibi_slopes * tf.cast(relative_positions, tf.float32)  # [num_heads, seq_len, seq_len]
        alibi_bias = tf.expand_dims(alibi_bias, axis=0)  # [1, num_heads, seq_len, seq_len]
        alibi_bias = tf.tile(alibi_bias, [batch_size, 1, 1, 1])  # [batch_size, num_heads, seq_len, seq_len]

        # Combine causal mask and ALiBi biases
        attention_mask = causal_mask + alibi_bias

        # Add the attention mask to the scaled attention logits
        scaled_attention_logits += attention_mask

        # Softmax to get attention weights
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        # Apply dropout to attention weights
        attention_weights = self.dropout_layer(attention_weights, training=training)

        # Compute the attention output
        attention_output = tf.matmul(attention_weights, value)

        # Concatenate heads
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.num_heads * self.key_dim))

        # Final linear layer
        output = self.output_dense(concat_attention)

        return output


    def get_config(self):
        config = super(AlibiMultiHeadAttention, self).get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "model_dim": self.model_dim,
            "dropout": self.dropout,
        })
        return config



class ReconstructCallback(Callback):
    def __init__(self, tokenizer: Tokenizer, shared_buffer: deque, max_input_length, num_samples=5, frequency=100):
        super().__init__()
        self.tokenizer = tokenizer
        self.shared_buffer = shared_buffer
        self.max_input_length = max_input_length
        self.num_samples = num_samples
        self.frequency = frequency

    def on_batch_end(self, batch, logs=None):
        if batch % self.frequency == 0 and len(self.shared_buffer) > 0:
            print(f"\nbatch {batch + 1} - Reconstruction samples:")
            for _ in range(self.num_samples):
                self.generate_sample()

    def generate_sample(self):
        # Sample a data point from the shared buffer
        data_point = random.choice(list(self.shared_buffer))

        query_type = data_point.get('type')

        if query_type not in query_type_to_query_fieldname:
            return  # Skip invalid sample

        query_fieldname = query_type_to_query_fieldname[query_type]
        query = data_point.get(query_fieldname, '').strip()
        result = json.dumps(data_point)

        if not query or not result:
            return  # Skip invalid samples

        # Construct the prompt
        prompt = f"{SPECIAL_START}{query}{SPECIAL_SEP}"
        input_indices = self.tokenizer.tokenize(prompt)

        print("\nprompt:")
        print(self.tokenizer.untokenize(input_indices))

        print("\nExpected result:")
        print(result)

        generated_text = self.generate_continuation(input_indices)
        print("\nModel prediction:")
        print(generated_text)

        print("-" * 50)

    def generate_continuation(self, input_indices):
        generated_indices = []
        current_input = input_indices.copy()

        for step in range(self.max_input_length):
            # Get model prediction
            prediction = self.model.predict(tf.expand_dims(current_input, 0), verbose=0)
            next_char_index = tf.argmax(prediction[0, -1, :]).numpy()
            next_char = self.tokenizer.index_to_char.get(next_char_index, 'UNKNOWN')

            # Add the predicted index to generated indices
            generated_indices.append(next_char_index)

            # Check for end token
            if next_char == self.tokenizer.index_to_char[self.tokenizer.char_to_index[SPECIAL_END]]:
                break

            # Update current input by appending the next predicted character
            current_input = np.concatenate([current_input, [next_char_index]])[-self.max_input_length:]

        # Once the loop finishes, convert the generated indices to a full string
        full_output_string = self.tokenizer.untokenize(generated_indices)
        return full_output_string




def verify_batch_generator(batch_generator, tokenizer, num_samples=5):
    print("\nVerifying batch generator by printing {} random examples...".format(num_samples))
    
    # Fetch one batch
    x_batch, y_batch = next(batch_generator)
    
    # Choose `num_samples` random indices from the batch
    sample_indices = random.sample(range(x_batch.shape[0]), num_samples)
    
    for idx in sample_indices:
        input_sequence = x_batch[idx]
        target_sequence = y_batch[idx]
        
        # Convert indices back to characters
        input_text = tokenizer.untokenize(input_sequence)
        target_text = tokenizer.untokenize(target_sequence)
        
        print("\nSample {}:".format(idx + 1))
        print("Input sequence (text):", input_text)
        print("Target sequence (text):", target_text)
        print("-" * 50)


def calculate_steps_per_epoch(jsonl_files, tokenizer, cache_file='dataset_size_cache.json'):
    """
    Calculate steps per epoch by counting the total number of samples generated by the batch generator.
    The function checks for cached results to avoid recalculating on subsequent runs.
    """
    # Compute a unique cache key based on the JSONL files and tokenizer settings
    cache_key = f"files_{hashlib.md5(str(jsonl_files).encode()).hexdigest()}_maxlen_{tokenizer.max_input_length}"

    # Load cache if available
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
            if cache_key in cached_data:
                total_samples = cached_data[cache_key]
                print(f"Using cached total samples: {total_samples}")
                return total_samples

    total_samples = 0

    # Process each file and calculate the total number of samples
    for jsonl_file in tqdm(jsonl_files, desc="Calculating total samples", unit="file"):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                query_type = data.get('type')

                if query_type not in ['movie', 'tvSeries', 'person', 'character']:
                    logging.debug(f"Skipping data point due to unknown query type: {query_type}")
                    continue

                query_fieldname = query_type_to_query_fieldname[query_type]
                query = data.get(query_fieldname, '').strip()

                result = json.dumps(data)

                # Full sequence including special characters
                full_text = f"{SPECIAL_START}{query}{SPECIAL_SEP}{result}{SPECIAL_END}"
                full_sequence = tokenizer.tokenize(full_text, pad_to_max_length=False)
                sequence_length = len(full_sequence)

                if sequence_length < 2:
                    continue  # Skip sequences that are too short

                # Number of samples for this data point
                num_samples = sequence_length - 1
                total_samples += num_samples

    # Cache the result for future use
    cached_data = {}
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
    cached_data[cache_key] = total_samples
    with open(cache_file, 'w') as f:
        json.dump(cached_data, f)

    print(f"Total samples across all JSONL files: {total_samples}")

    return total_samples

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
                # format = "x minutes ago"
                last_modified = datetime.fromtimestamp(Path(cache_file).stat().st_mtime)
                last_modified_ago = datetime.now() - last_modified
                print(f"Loading alphabet from cache. Last modified {last_modified_ago}")
                return cache_data['alphabet']

    # Initialize a set to collect all unique characters
    alphabet_set = set()

    # Process each JSONL file to compute the alphabet
    for jsonl_path in jsonl_files:
        with open(jsonl_path, 'r', encoding='utf-8') as jsonl_file:
            for line in jsonl_file:
                alphabet_set.update(line)

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


class CommitChangesToGitCallback(Callback):
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


class TensorBoardBatchLogger(Callback):
    def __init__(self, log_dir, update_freq=100):
        super().__init__()
        self.log_dir = log_dir
        self.update_freq = update_freq
        self.writer = tf.summary.create_file_writer(log_dir)
        self.step = 0

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        # Add the current learning rate to the logs
        lr = self.model.optimizer.learning_rate.numpy()
        logs.update({'learning_rate': lr})
        
        if self.step % self.update_freq == 0:
            with self.writer.as_default():
                for name, value in logs.items():
                    tf.summary.scalar(name, value, step=self.step)
                self.writer.flush()
        self.step += 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    main()
