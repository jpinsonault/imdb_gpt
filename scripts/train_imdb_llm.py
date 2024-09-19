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

tf_version = tf.__version__

SPECIAL_PAD = '\u200C'
SPECIAL_START = '\u200D'
SPECIAL_END = '\u200E'
SPECIAL_SEP = '\u200F'


def main(args):
    character_llm_training()

def character_llm_training():
    # Initialize paths and configuration
    data_dir = Path(project_config['data_dir'])
    input_length = project_config['llm']['input_length']
    batch_size = project_config['llm']['batch_size']
    character_embedding_dim = project_config['llm']['character_embedding_dim']
    num_epochs = project_config['llm']['epochs']
    jsonl_files = [data_dir / jsonl_file for jsonl_file in project_config['dataset']['jsonl_files']]
    
    steps_per_epoch = calculate_steps_per_epoch(jsonl_files, input_length, batch_size)

    print_project_config(project_config)

    tokenizer = Tokenizer.from_files(jsonl_files, input_length)
    model = kermit_language_model(
        input_length=input_length,
        alphabet_size=tokenizer.alphabet_size,
        character_embedding_dim=character_embedding_dim
    )
    
    dataset = autoencoder_batch_generator(jsonl_files, batch_size, tokenizer)
    verify_batch_generator(dataset, tokenizer)
    callbacks = get_callbacks(tokenizer, jsonl_files, input_length, steps_per_epoch, num_epochs)
    
    # Compile and train model
    compile_and_train_model(model, dataset, steps_per_epoch, num_epochs, callbacks)


def get_callbacks(tokenizer, jsonl_files, input_length, steps_per_epoch, num_epochs):
    reconstruct_callback = ReconstructCallback(
        tokenizer=tokenizer,
        max_input_length=input_length,
        num_samples=5,
        frequency=100,
        jsonl_files=jsonl_files
    )

    logs_dir = create_logs_dir()
    tensorboard_callback = TensorBoardBatchLogger(log_dir=str(logs_dir), update_freq=5)
    commit_changes_to_git_callback = CommitChangesToGitCallback(logs_dir)

    return [
        commit_changes_to_git_callback,
        tensorboard_callback,
        reconstruct_callback,
        keras.callbacks.TensorBoard('./logs', update_freq=5)
    ]


def compile_and_train_model(model, dataset, steps_per_epoch, num_epochs, callbacks):
    initial_learning_rate = 0.0002
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=steps_per_epoch * num_epochs,
        alpha=0.0
    )

    print("\n> Compiling model\n")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='sparse_categorical_crossentropy',
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


def create_logs_dir():
    now_int = int(datetime.now().timestamp())
    folder_suggestion = f"{now_int}_{suggest_folder_name()}"
    logs_dir = Path("logs") / folder_suggestion
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


class Tokenizer:
    def __init__(self, char_to_index, max_input_length):
        self.char_to_index = char_to_index
        self.alphabet_size = len(char_to_index)
        self.index_to_char = {index: char for char, index in char_to_index.items()}
        self.max_input_length = max_input_length

    @classmethod
    def from_files(cls, jsonl_files, max_input_length):
        alphabet = get_alphabet(jsonl_files) + [SPECIAL_PAD, SPECIAL_START, SPECIAL_END, SPECIAL_SEP]
        char_to_index = {char: index for index, char in enumerate(sorted(alphabet))}
        return cls(char_to_index, max_input_length)

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
    


logging.basicConfig(level=logging.ERROR)

def autoencoder_batch_generator(jsonl_files, batch_size, tokenizer: Tokenizer):
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
        data_indices[dataset_choice] += 1

        query = data_point.get('query', '').strip()
        result = data_point.get('result', '').strip()

        if not query or not result:
            logging.debug("Skipping data point due to empty query or result")
            continue

        # Build input and target sequences
        input_prefix = f"{SPECIAL_START}{query}{SPECIAL_SEP}"
        input_prefix_indices = tokenizer.tokenize(input_prefix, pad_to_max_length=False)
        result_indices = tokenizer.tokenize(result + SPECIAL_END, pad_to_max_length=False)

        # Build full sequence
        full_sequence = input_prefix_indices + result_indices

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


class ReduceSum(tf.keras.layers.Layer):
    def __init__(self, axis, keepdims=False, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=self.axis, keepdims=self.keepdims)
    
class Stack(tf.keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.stack(inputs, axis=self.axis)
    
class Squeeze(tf.keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)
    
    def compute_output_shape(self, input_shape):
        return input_shape[:self.axis] + input_shape[self.axis + 1:]


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

def create_combined_positional_mask(input_sequence, positional_encoding):
    token_dim = input_sequence.shape[-1]
    last_token = input_sequence[:, -1, :]

    input_length = input_sequence.shape[1]

    expert_weights = create_experts(token_dim, input_length, num_experts=16)

    last_token_positional_mask = moe_self_attention(input_vector=last_token, hidden_dim=token_dim, expert_weights=expert_weights, output_dim=input_length, activation=None)
    last_token_positional_mask = Reshape((input_length, 1), name=n("reshape_last_token_positional_mask_2"))(last_token_positional_mask)

    tokenwise_positional_mask = moe_1x1_projection(input_dim=token_dim, output_dim=1, num_experts=16, activation=None)(input_sequence)

    positional_mask = Softmax(axis=1, name=n("softmax_positional_mask"))(tokenwise_positional_mask + last_token_positional_mask)

    return positional_mask

def create_global_token(input_sequence, token_dim):
    query_tokens = Conv1D(filters=token_dim, kernel_size=1, padding='same', activation=None, name=n("conv_query_tokens"))(input_sequence)
    global_token = ReduceSum(axis=1, keepdims=True, name=n("reduce_sum_global_token"))(query_tokens)
    
    return global_token


def dynamic_1x1_convolution(input_tokens, context_token, output_dim):
    input_dim = input_tokens.shape[-1]

    conv_weights = Dense(input_dim * output_dim, activation=None, name=n("dense_conv_weights"))(context_token)
    conv_weights = Reshape((input_dim, output_dim), name=n("reshape_conv_weights"))(conv_weights)

    def dynamic_einsum(x):
        input_tokens, conv_weights = x
        return tf.einsum('bij,bjk->bik', input_tokens, conv_weights)
    
    return Lambda(dynamic_einsum, name=n("lambda_einsum"), output_shape=(input_tokens.shape[1], output_dim))([input_tokens, conv_weights])


class CustomDense(Layer):
    def call(self, args):
        inputs = args[0]
        weight_matrix = args[1]
        return tf.matmul(inputs, weight_matrix)


def dynamic_self_attention(input_vector, hidden_dim):
    vector_dim = input_vector.shape[-1]
    
    def generate_weights(input_vec):
        weights = Dense(vector_dim * vector_dim, activation=None, name=n("dense_weights"))(input_vec)
        return weights
    
    squeezed_input = Dense(hidden_dim, activation=None, name=n("dense_squeezed_input"))(input_vector)

    weight_matrix = generate_weights(squeezed_input)

    weight_matrix = Reshape((vector_dim, vector_dim))(weight_matrix)

    reshaped_input = Reshape((1, vector_dim))(input_vector)
    output = Lambda(lambda x: tf.matmul(x[0], x[1]), name=n("lambda_matmul_attention"))([reshaped_input, weight_matrix])
    output = Reshape((vector_dim,), name=n("self_attention_output"))(output)

    return output

def attention_cross_attention(input_vector, context_vector, hidden_dim):
    vector_dim = input_vector.shape[-1]
    
    def generate_weights(context_vec):
        weights = Dense(vector_dim * vector_dim, activation=None, name=n("dense_weights"))(context_vec)
        return weights
    
    reduced_context = Dense(hidden_dim, activation=None, name=n("dense_reduced_context"))(context_vector)
    weight_matrix = generate_weights(reduced_context)

    weight_matrix = Reshape((vector_dim, vector_dim))(weight_matrix)

    reshaped_input = Reshape((1, vector_dim))(input_vector)
    output = Lambda(lambda x: tf.matmul(x[0], x[1]), name=n("lambda_matmul_attention"))([reshaped_input, weight_matrix])
    output = Reshape((vector_dim,), name=n("attention_output"))(output)

    return output


def additive_self_attention(character_embedding_dim, num_heads, experts, positional_encoding, activation=None):
    def apply_attention(input_sequence):
        head_dim = character_embedding_dim // num_heads

        if num_heads == 1:
            split_inputs = [input_sequence]
        else:
            split_inputs = Lambda(lambda x: tf.split(x, num_heads, axis=-1), name=n("lambda_split_inputs"))(input_sequence)

        head_outputs = []
        for i in range(num_heads):
            head_input_tokens = split_inputs[i]
            tf.print(f"head_input_tokens: {head_input_tokens.shape}")

            input_positional_mask = create_combined_positional_mask(head_input_tokens, positional_encoding=positional_encoding)            

            masked_input_tokens = head_input_tokens * input_positional_mask

            global_query = create_global_token(masked_input_tokens, head_dim)

            global_query = moe_self_attention(global_query, head_dim, experts, head_dim, activation=activation)
            global_query = Reshape((1, head_dim), name=n("reshape_global_query"))(global_query)

            queried_tokens = masked_input_tokens * global_query
            
            output_tokens = Conv1D(filters=head_dim, kernel_size=1, padding='same', activation=None, name=n("conv_attention"))(queried_tokens)
            head_outputs.append(output_tokens)

        head_outputs = Concatenate(name=n("concatenate_heads"))(head_outputs)

        output_tokens = Add(name=n("add_and_norm_attention"))([head_outputs, input_sequence])
    
        return output_tokens
    
    return apply_attention


def ff_layer(character_embedding_dim, activation):
    def apply_ff_layer(input_sequence):
        x = Dense(character_embedding_dim * 4, activation=activation, name=n("dense_ff_1"))(input_sequence)
        x = Conv1D(filters=character_embedding_dim, kernel_size=1, padding='same', activation=None, name=n("conv_ff_2"))(x)
        x = Add(name=n("add_ff"))([x, input_sequence])
        x = LayerNormalization(name=n("layer_norm_ff"))(x)
        return x

    return apply_ff_layer


def moe_1x1_projection(input_dim, output_dim, num_experts, activation=None):
    # Stack the expert weights into a single tensor of shape (num_experts, D, D)
    expert_weights = [tf.Variable(tf.random.normal([input_dim, output_dim]), name=n(f"expert_{i}_weight")) for i in range(num_experts)]
    expert_weights = Stack(axis=0, name=n("stack_expert_weights"))(expert_weights)

    # Expand dimensions of expert_weights for broadcasting
    expert_weights = Lambda(lambda x: tf.expand_dims(x, axis=0), name=n("expand_expert_weights"))(expert_weights)
    
    def apply_moe_layer(input_sequence):
        input_length = input_sequence.shape[1]

        last_token = input_sequence[:, -1, :]

        expert_scores = Dense(input_dim, activation='leaky_relu', name=n("dense_last_token"))(last_token)
        expert_scores += last_token

        expert_scores = Dense(num_experts, activation=None, name=n("expert_score_projection"))(last_token)
        
        # Softmax the scores
        expert_scores_softmax = Softmax(name=n("expert_score_softmax"))(expert_scores)
        
        # Reshape the softmax scores to match the shape needed for multiplication (batch_size, num_experts, 1)
        expert_scores_softmax = Reshape((num_experts, 1, 1), name=n("reshape_scores"))(expert_scores_softmax)

        # Multiply the expert weight matrices by the scores and sum them up
        combined_weight_matrix = Lambda(lambda x: tf.reduce_sum(x[0] * x[1], axis=1),
                                        output_shape=(input_dim, output_dim),
                                        name=n("combine_weights"))([expert_scores_softmax, expert_weights])
        
        # Apply the combined weight matrix as the 1x1 projection for each token
        output = Lambda(lambda x: tf.matmul(x[0], x[1]), 
                        output_shape=(input_length, output_dim),
                        name=n("lambda_matmul_projection"))([input_sequence, combined_weight_matrix])
        
        if activation is not None:
            output = Activation(activation, name=n("moe_activation"))(output)
        
        return output

    return apply_moe_layer


def moe_self_attention(input_vector, hidden_dim, expert_weights, output_dim, activation=None):
    """Applies a Mixture of Experts (MoE) self-attention mechanism to the input vector. 

    The function projects the input vector to a set of scores over a group of expert weight matrices.
    These scores are used to combine the expert matrices dynamically, creating a custom weight matrix for each input vector.
    The input vector is then projected to the output dimension using this dynamically generated weight matrix.
    """
    num_experts = expert_weights.shape[0]
    input_dim = input_vector.shape[-1]

    # Expand dimensions of expert_weights for broadcasting
    expert_weights = Lambda(lambda x: tf.expand_dims(x, axis=0), name=n("expand_expert_weights"))(expert_weights)
    
    def generate_weights(input_vec):
        # Project the input vector to a set of scores for the experts
        expert_scores = Dense(input_dim, activation='leaky_relu', name=n("expert_score_projection"))(input_vec)
        expert_scores += input_vec

        expert_scores = Dense(num_experts, activation=None, name=n("expert_score_projection"))(expert_scores)

        expert_scores_softmax = Softmax(name=n("expert_score_softmax"))(expert_scores)
        
        # Reshape the softmax scores to match the shape needed for multiplication (batch_size, num_experts, 1, 1)
        expert_scores_softmax = Lambda(lambda x: tf.reshape(x, (-1, num_experts, 1, 1)), name=n("reshape_scores"))(expert_scores_softmax)
        
        # Multiply and sum using broadcasting
        combined_weight_matrix = Lambda(lambda x: tf.reduce_sum(x[0] * x[1], axis=1),
                                        output_shape=(hidden_dim, output_dim),
                                        name=n("combine_weights"))([expert_scores_softmax, expert_weights])
        
        return combined_weight_matrix
    
    # Squeeze the input vector to a hidden dimension
    squeezed_input = Dense(hidden_dim, activation=None, name=n("dense_squeezed_input"))(input_vector)

    # Generate the dynamic weight matrix using the MoE approach
    weight_matrix = generate_weights(squeezed_input)

    # Reshape the input vector for matrix multiplication
    reshaped_input = Reshape((1, hidden_dim))(squeezed_input)
    
    # Apply the dynamic weight matrix to the input vector
    output = Lambda(lambda x: tf.matmul(x[0], x[1]), 
                    output_shape=(1, output_dim),
                    name=n("lambda_matmul_attention"))([reshaped_input, weight_matrix])
    
    if activation is not None:
        output = Activation(activation, name=n("self_attention_activation"))(output)
    
    output = Reshape((output_dim,), name=n("self_attention_output"))(output)

    return output


def create_experts(input_dim, output_dim, num_experts):
    expert_weights = [tf.Variable(tf.random.normal([input_dim, output_dim]), name=f"expert_{i}_weight") for i in range(num_experts)]
    expert_weights = Stack(axis=0, name=n("stack_expert_weights"))(expert_weights)
    
    return expert_weights


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


def transformer_decoder_block(input, num_heads, ff_dim, dropout_rate=0.1):
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

######################################################################################################################################
#-------------------------------------------------------------------------------------------------------------------------------------
#.....................................................................................................................................

def kermit_language_model(input_length, alphabet_size, character_embedding_dim, dropout_rate=0.001):
    num_decoder_blocks = 4
    num_heads = 1

    inputs = Input(shape=(input_length,), dtype=tf.int32)
    embedding = Embedding(input_dim=alphabet_size, output_dim=character_embedding_dim)(inputs)
    embedding = layers.Dropout(dropout_rate)(embedding)

    reduced_to_half = Conv1D(character_embedding_dim, 3, strides=2, padding='same', activation='gelu')(embedding)
    reduced_to_quarter = Conv1D(character_embedding_dim, 3, strides=2, padding='same', activation='gelu')(reduced_to_half)

    reduced_to_quarter = LayerNormalization()(reduced_to_quarter)

    x = reduced_to_quarter

    # Decoder blocks
    for _ in range(num_decoder_blocks):
        x = transformer_decoder_block(x, num_heads, ff_dim=character_embedding_dim, dropout_rate=dropout_rate)

    back_to_half = Conv1DTranspose(character_embedding_dim, 3, strides=2, padding='same', activation='gelu')(x)
    back_to_full = Conv1DTranspose(character_embedding_dim, 3, strides=2, padding='same', activation='gelu')(back_to_half)

    # Final output layer
    outputs = Dense(alphabet_size, activation='softmax')(back_to_full)

    model = Model(inputs=inputs, outputs=outputs, name="transformer_autoencoder_with_combined_attention_pooling")
    return model

#......................................................................................................................................
#-------------------------------------------------------------------------------------------------------------------------------------
######################################################################################################################################


class ReconstructCallback(Callback):
    def __init__(self, tokenizer: Tokenizer, max_input_length, num_samples=5, frequency=100, jsonl_files=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.num_samples = num_samples
        self.frequency = frequency
        self.jsonl_files = jsonl_files

    def on_batch_end(self, batch, logs=None):
        if batch % self.frequency == 0:
            print(f"\nbatch {batch + 1} - Reconstruction samples:")
            for _ in range(self.num_samples):
                self.generate_sample()

    def generate_sample(self):
        # Randomly select a JSONL file and a line from it
        jsonl_file = random.choice(self.jsonl_files)
        with open(jsonl_file, 'r') as f:
            lines = f.readlines()
            line = random.choice(lines)
            data = json.loads(line)

        query = data['query']
        result = data['result']

        # Construct the prompt
        prompt = f"{SPECIAL_START}{query}{SPECIAL_SEP}"  # query + SPECIAL_SEP
        input_indices = self.tokenizer.tokenize(prompt)

        print("\nInput query:")
        print(self.tokenizer.untokenize(input_indices))

        print("\nExpected result:")
        print(result)

        generated_text = self.generate_continuation(input_indices)
        print("\nModel prediction:")
        print(generated_text)

        print("-" * 50)

    def generate_continuation(self, input_indices):
        generated_indices = []  # Store all predicted indices
        current_input = input_indices.copy()

        for step in range(self.max_input_length):
            # Get the model's prediction for the current input
            prediction = self.model.predict(tf.expand_dims(current_input, 0), verbose=0)

            # Extract the next character to generate
            next_char_index = tf.argmax(prediction[0, -1, :]).numpy()
            next_char = self.tokenizer.index_to_char.get(next_char_index, 'UNKNOWN')

            # Add the predicted index to the list
            generated_indices.append(next_char_index)

            if next_char == SPECIAL_END:
                break

            # Update the input for the next prediction step
            current_input = tf.concat([current_input, [next_char_index]], axis=0)[-self.max_input_length:]

        # Convert the full list of indices back to characters once more for the final output
        full_output_string = self.tokenizer.untokenize(generated_indices)

        # Return the final readable text as well
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
        print("Input sequence (indices):", input_sequence)
        print("Input sequence (text):", input_text)
        print("Target sequence (indices):", target_sequence)
        print("Target sequence (text):", target_text)
        print("-" * 50)


def calculate_steps_per_epoch(jsonl_files, max_input_length, batch_size, cache_file='dataset_size_cache.json'):
    """
    Calculate steps per epoch by counting the number of possible windows in the dataset.
    The function checks for cached results to avoid recalculating on subsequent runs.
    """
    cache_key = f"files_{hashlib.md5(str(jsonl_files).encode()).hexdigest()}"
    
    # Load cache if available
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
            if cache_key in cached_data:
                total_windows = cached_data[cache_key]
                print(f"Using cached total windows: {total_windows}")
                return total_windows // batch_size

    total_windows = 0

    # Process each file and calculate the total number of windows
    for jsonl_file in tqdm(jsonl_files, desc="Calculating windows", unit="file"):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                query = data.get('query', '').strip()
                result = data.get('result', '').strip()
                if not query or not result:
                    continue

                # Full sequence including special characters
                full_text = f"{SPECIAL_START}{query}{SPECIAL_SEP}{result}{SPECIAL_END}"
                text_length = len(full_text)
                
                # Calculate the number of possible windows of length max_input_length
                line_windows = max(1, text_length - max_input_length + 1)
                total_windows += line_windows

    # Cache the result for future use
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
    else:
        cached_data = {}

    # Store the total windows count in the cache
    cached_data[cache_key] = total_windows
    with open(cache_file, 'w') as f:
        json.dump(cached_data, f)

    steps_per_epoch = total_windows // batch_size

    print(f"Total windows across all JSONL files: {total_windows}")
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {steps_per_epoch}")

    return steps_per_epoch


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

    main(args)
