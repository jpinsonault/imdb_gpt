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
    data_dir = Path(project_config['data_dir'])
    input_length = project_config['llm']['input_length']
    batch_size = project_config['llm']['batch_size']
    character_embedding_dim = project_config['llm']['character_embedding_dim']
    num_epochs = project_config['llm']['epochs']
    jsonl_files = [data_dir / jsonl_file for jsonl_file in project_config['dataset']['jsonl_files']]

    print_project_config(project_config)

    alphabet = get_alphabet(jsonl_files)
    alphabet += [SPECIAL_PAD, SPECIAL_START, SPECIAL_END, SPECIAL_SEP]
    print(f"Alphabet size: {len(alphabet)}")

    char_to_index = {char: index for index, char in enumerate(sorted(alphabet))}
    index_to_char = {index: char for char, index in char_to_index.items()}

    model = kermit_language_model(
        input_length=input_length,
        alphabet_size=len(alphabet),
        character_embedding_dim=character_embedding_dim
    )
    
    tokenizer = Tokenizer(char_to_index=char_to_index, max_input_length=input_length)

    # Create a generator for the JSONL files
    dataset = autoencoder_batch_generator(jsonl_files, batch_size, tokenizer)

    reconstruct_callback = ReconstructCallback(
        char_to_index=char_to_index,
        index_to_char=index_to_char,
        max_input_length=input_length,
        num_samples=5,
        frequency=100,
        jsonl_files=jsonl_files
    )

    print("\n> compiling model\n")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=linear_mask_loss,
        metrics=[NextTokenAccuracy()],
    )

    model.summary()

    now_int = int(datetime.now().timestamp())
    folder_suggestion = suggest_folder_name()
    folder_suggestion = f"{now_int}_{folder_suggestion}"
    logs_dir = Path("logs") / folder_suggestion
    logs_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_callback = TensorBoardBatchLogger(log_dir=str(logs_dir), update_freq=5)

    commit_changes_to_git_callback = CommitChangesToGitCallback(logs_dir)

    learning_rate_callback = ScheduledLearningRateCallback(
        schedule=[
            (0, 0.0002),
            (1, 0.0001),
            (2, 0.00005),
            (3, 0.00002),
            (4, 0.00001),
            (5, 0.000005),
            (6, 0.000002),
        ]
    )

    steps_per_epoch = calculate_steps_per_epoch(jsonl_files, input_length, batch_size)

    model.fit(
        dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        callbacks=[
            learning_rate_callback,
            commit_changes_to_git_callback,
            tensorboard_callback,
            reconstruct_callback,
            keras.callbacks.TensorBoard('./logs', update_freq=5)
        ]
    )


def autoencoder_batch_generator(jsonl_files, batch_size, tokenizer):
    """
    Efficient batch generator for autoencoder training using multiple JSONL files.
    Uses character-level tokenization with special Unicode characters for tokens.
    """
    
    # Ensure special tokens are in the tokenizer
    for token in [SPECIAL_PAD, SPECIAL_START, SPECIAL_END, SPECIAL_SEP]:
        if token not in tokenizer.char_to_index:
            tokenizer.add_token(token)
    
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
    x_batch = np.zeros((batch_size, tokenizer.max_input_length), dtype=np.int32)
    y_batch = np.zeros((batch_size, tokenizer.max_input_length, len(tokenizer.char_to_index)), dtype=np.float32)

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
            continue

        # Combine query and result with special separator
        full_text = f"{query}{SPECIAL_SEP}{result}{SPECIAL_END}"

        try:
            # Tokenize the full text
            input_chars = list(full_text)
            input_token_indices = [tokenizer.char_to_index.get(char, tokenizer.char_to_index[SPECIAL_PAD]) for char in input_chars]

            # Prepare input (query + separator) and output (result + end token)
            query_sep_length = len(query) + 1  # +1 for the separator
            input_indices = input_token_indices[:query_sep_length]
            output_indices = input_token_indices[query_sep_length:]

            # Pad if necessary
            if len(input_indices) < tokenizer.max_input_length:
                pad_length = tokenizer.max_input_length - len(input_indices)
                input_indices += [tokenizer.char_to_index[SPECIAL_PAD]] * pad_length
                output_indices += [tokenizer.char_to_index[SPECIAL_PAD]] * pad_length

            # Truncate if necessary
            input_indices = input_indices[:tokenizer.max_input_length]
            output_indices = output_indices[:tokenizer.max_input_length]

            # Update the batch
            x_batch[batch_index] = input_indices
            
            # One-hot encode the output
            for i, token_index in enumerate(output_indices):
                y_batch[batch_index, i, token_index] = 1

            batch_index += 1

            # Yield a full batch
            if batch_index == batch_size:
                yield x_batch, y_batch
                # Reset for next batch
                batch_index = 0
                x_batch = np.zeros((batch_size, tokenizer.max_input_length), dtype=np.int32)
                y_batch = np.zeros((batch_size, tokenizer.max_input_length, len(tokenizer.char_to_index)), dtype=np.float32)

        except Exception as e:
            traceback.print_exc()
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"{exc_type} {exc_obj} {exc_tb.tb_lineno}")
            print(f"\nFailed to process entity: {query}\n")
        
        # Check if all datasets are exhausted
        if all(index >= size for index, size in zip(data_indices, dataset_sizes)):
            # Reset all indices and continue
            data_indices = [0] * len(jsonl_files)


def linear_mask_loss(y_true, y_pred, mask_start=0.25, mask_end=1.0):
    # Get the shape of the predictions
    sequence_length = tf.shape(y_pred)[1]
    
    # Create a linear mask that decreases from mask_start to mask_end
    mask = tf.linspace(mask_start, mask_end, sequence_length)
    
    # Expand the mask to match the shape of y_pred
    mask = tf.reshape(mask, [1, sequence_length])
    
    # Apply the mask to the loss
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    masked_loss = loss * mask
    
    # Return the mean loss, with the mask applied
    return tf.reduce_mean(masked_loss)


class Tokenizer:
    def __init__(self, char_to_index, max_input_length):
        self.char_to_index = char_to_index
        self.index_to_char = {index: char for char,
                              index in char_to_index.items()}
        self.index_to_char[char_to_index[SPECIAL_PAD]] = '@'
        self.index_to_char[char_to_index[SPECIAL_START]] = '^'
        self.index_to_char[char_to_index[SPECIAL_END]] = '$'
        self.index_to_char[char_to_index[SPECIAL_SEP]] = '#'
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
    

class NextTokenAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='next_token_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Focus on the last token in the sequence
        y_true_last = tf.argmax(y_true[:, -1, :], axis=-1)
        y_pred_last = tf.argmax(y_pred[:, -1, :], axis=-1)
        
        # Calculate the accuracy for the last token
        values = tf.cast(tf.equal(y_true_last, y_pred_last), tf.float32)
        
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            values = tf.multiply(values, sample_weight)
        
        self.total.assign_add(tf.reduce_sum(values))
        self.count.assign_add(tf.cast(tf.size(y_true_last), tf.float32))

    def result(self):
        return self.total / self.count

    def reset_state(self):
        self.total.assign(0.)
        self.count.assign(0.)


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

class LinformerSelfAttention(Layer):
    def __init__(self, embed_dim, num_heads, seq_len, k=64, dropout=0.1):
        super(LinformerSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.k = k
        self.dropout = dropout

        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"

        self.head_dim = embed_dim // num_heads

        # Linear projections for queries, keys, and values
        self.query = Dense(embed_dim)
        self.key = Dense(embed_dim)
        self.value = Dense(embed_dim)

        # Low-rank projection matrices
        self.proj_k = Dense(self.head_dim)  # Adjust projection to match head_dim
        self.proj_v = Dense(self.head_dim)  # Adjust projection to match head_dim

        # Output linear layer
        self.out = Dense(embed_dim)
        self.dropout = Dropout(dropout)

    def build(self, input_shape):
        # No additional weights are needed here, but this method marks the layer as built.
        super(LinformerSelfAttention, self).build(input_shape)

    def call(self, x, mask=None):
        batch_size = tf.shape(x)[0]

        # Project input to queries, keys, and values
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Split the embedding into multiple heads
        Q = tf.reshape(Q, (batch_size, self.seq_len, self.num_heads, self.head_dim))
        K = tf.reshape(K, (batch_size, self.seq_len, self.num_heads, self.head_dim))
        V = tf.reshape(V, (batch_size, self.seq_len, self.num_heads, self.head_dim))

        # Apply low-rank projection
        K = self.proj_k(K)
        V = self.proj_v(V)

        # Reshape K and V to align with Q for matmul
        Q = tf.transpose(Q, perm=[0, 2, 1, 3])  # [batch_size, num_heads, seq_len, head_dim]
        K = tf.transpose(K, perm=[0, 2, 1, 3])  # [batch_size, num_heads, k, head_dim]
        V = tf.transpose(V, perm=[0, 2, 1, 3])  # [batch_size, num_heads, k, head_dim]

        # Compute attention scores
        attention_scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))

        if mask is not None:
            attention_scores += (mask * -1e9)

        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_weights = self.dropout(attention_weights)

        # Combine heads
        attention_output = tf.matmul(attention_weights, V)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])  # Back to original shape
        attention_output = tf.reshape(attention_output, (batch_size, self.seq_len, self.embed_dim))

        # Final linear projection
        output = self.out(attention_output)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.seq_len, self.embed_dim)

def linformer_language_model(input_length, alphabet_size, character_embedding_dim, num_blocks=4, num_heads=4, k=64):
    inputs = tf.keras.Input(shape=(input_length,), dtype=tf.int32, name="input_layer")
    embedding_layer = Embedding(input_dim=alphabet_size, output_dim=character_embedding_dim, name="embedding_layer")
    embedding = embedding_layer(inputs)
    
    x = embedding
    for i in range(num_blocks):
        attention_output = LinformerSelfAttention(
            embed_dim=character_embedding_dim,
            num_heads=num_heads,
            seq_len=input_length,
            k=k
        )(x)
        x = Add()([x, attention_output])
        x = LayerNormalization()(x)
        x = Dense(character_embedding_dim, activation='relu')(x)
    
    # The final Dense layer should output a probability distribution over the alphabet for each token
    x = Dense(alphabet_size, activation='softmax', name="output_layer")(x)

    model = tf.keras.Model(inputs=inputs, outputs=x, name="LinformerLanguageModel")
    return model


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


def print_debug_batches(dataset, index_to_char, num_batches=5, samples_per_batch=3):
    """
    Print debug information for a specified number of batches from the generator.
    
    Args:
    dataset: The TensorFlow dataset
    index_to_char: Dictionary mapping indices to characters
    num_batches: Number of batches to print (default: 5)
    samples_per_batch: Number of samples to print from each batch (default: 3)
    """
    batch_iterator = iter(dataset)
    
    for batch_num in range(num_batches):
        print(f"\n{'='*40}\nBatch {batch_num + 1}\n{'='*40}")
        
        try:
            batch = next(batch_iterator)
            inputs, targets = batch
            
            for sample in range(min(samples_per_batch, len(inputs))):
                print(f"\nSample {sample + 1}:")
                input_sequence = inputs[sample].numpy()
                target = targets[sample].numpy()
                
                print("Input sequence:")
                print(input_sequence)
                print("\nInput decoded:")
                print(''.join([index_to_char[idx] for idx in input_sequence]))
                
                print("\nTarget:")
                print(target)
                print("\nTarget decoded:")
                target_index = np.argmax(target)
                print(f"Index: {target_index}, Character: {index_to_char[target_index]}")
                
                print("\n" + "-"*30)
        
        except StopIteration:
            print("Reached the end of the dataset.")
            break


class ReconstructCallback(keras.callbacks.Callback):
    def __init__(self, char_to_index, index_to_char, max_input_length, num_samples, frequency, jsonl_files, temperature=1.0):
        self.char_to_index = char_to_index
        self.index_to_char = index_to_char
        self.max_input_length = max_input_length
        self.num_samples = num_samples
        self.frequency = frequency
        self.jsonl_files = jsonl_files
        self.temperature = temperature

    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch + 1} Reconstruction Outputs:\n")
        self._print_predictions()

    def on_batch_end(self, batch, logs=None):
        if batch % self.frequency == 0:
            print(f"\nBatch {batch} Reconstruction Outputs:\n")
            self._print_predictions()

    def _get_random_sequence(self):
        random_file = random.choice(self.jsonl_files)
        with open(random_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            random_line = random.choice(lines)
            data = json.loads(random_line)
            query = data.get('query', '').strip()
            result = data.get('result', '').strip()
            input_text = f"{query}{SPECIAL_SEP}"
            if len(input_text) > self.max_input_length:
                input_text = input_text[:self.max_input_length]
            return input_text, result
    
    def _tokenize(self, sequence):
        return np.array([self.char_to_index.get(char, self.char_to_index[SPECIAL_PAD]) for char in sequence], dtype=np.int32)

    def _select_top_character(self, predictions):
        return np.argmax(predictions, axis=-1)

    def _print_predictions(self):
        for _ in range(self.num_samples):
            input_sequence, target_result = self._get_random_sequence()
            input_tensor = self._tokenize(input_sequence)
            
            generated_sequence = list(input_sequence)
            color_coded_sequence = []
            
            for i in range(self.max_input_length - len(input_sequence)):
                try:
                    input_tensor_reshaped = np.reshape(input_tensor, (1, self.max_input_length))
                    predictions = self.model.predict(input_tensor_reshaped, verbose=0)[0]

                    # Focus on the prediction for the last token in the sequence
                    last_token_prediction = predictions[len(generated_sequence) - 1]

                    if i == 0:
                        top_5_indices = np.argsort(last_token_prediction)[-5:][::-1]
                        top_5_chars = [self.index_to_char[idx] for idx in top_5_indices]
                        top_5_probs = last_token_prediction[top_5_indices]
                        print(f"Step {i} - Top 5 predictions for the next token:")
                        for char, prob in zip(top_5_chars, top_5_probs):
                            bar_length = 40
                            bar_fill = int(prob * bar_length)
                            bar = '#' * bar_fill + '-' * (bar_length - bar_fill)
                            print(f"{char}: {prob:.4f} | {bar}")

                    predicted_index = self._select_top_character(last_token_prediction)
                    predicted_char = self.index_to_char[predicted_index]
                    confidence = last_token_prediction[predicted_index]
                    
                    color_coded_char = self._color_code(predicted_char, confidence)
                    color_coded_sequence.append(color_coded_char)
                    
                    # Append the predicted character to the generated sequence
                    generated_sequence.append(predicted_char)
                    
                    # Update input_tensor to reflect the new prediction
                    input_tensor = np.concatenate([input_tensor[1:], [predicted_index]])
                    
                    # Stop if we've reached the end token or the maximum length
                    if predicted_char == SPECIAL_END or len(generated_sequence) == self.max_input_length:
                        break
                    
                except Exception as e:
                    print(f"Error during prediction: {e}")
                    break
            
            print(f"Input    | {input_sequence}")
            print(f"Target   | {target_result}")
            print(f"Predicted| {''.join(generated_sequence[len(input_sequence):])}")
            print(f"Color    | {''.join(color_coded_sequence)}")
            print("\n")

    def _color_code(self, char, confidence):
        if confidence > 0.92:
            return f"\033[30;42m{char}\033[0m"  # Green background, black text
        elif confidence > 0.7:
            return f"\033[30;43m{char}\033[0m"  # Yellow background, black text
        else:
            return f"\033[30;41m{char}\033[0m"  # Red background, black text


def calculate_steps_per_epoch(jsonl_files, max_input_length, batch_size, cache_file='dataset_size_cache.json'):
    cache_key = f"files_{hashlib.md5(str(jsonl_files).encode()).hexdigest()}"
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
            if cache_key in cached_data:
                total_windows = cached_data[cache_key]
                print(f"Using cached total windows: {total_windows}")
                return total_windows // batch_size

    total_windows = 0

    for jsonl_file in tqdm(jsonl_files, desc="Calculating windows", unit="file"):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                query = data.get('query', '')
                result = data.get('result', '')
                full_text = f"{SPECIAL_START}{query}{SPECIAL_SEP}{result}{SPECIAL_END}"
                text_length = len(full_text)
                # Calculate the number of windows in this line
                line_windows = max(0, (text_length - max_input_length) + 1)
                total_windows += line_windows

    # Cache the result
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
    else:
        cached_data = {}
    
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


def are_models_same(model1, model2):
    """
    Compare two Keras models to check if they have the same architecture and weights.
    
    :param model1: First Keras model
    :param model2: Second Keras model
    :return: Boolean indicating if the models are the same
    """
    # Check if the models have the same number of layers
    if len(model1.layers) != len(model2.layers):
        return False
    
    # Compare each layer
    for layer1, layer2 in zip(model1.layers, model2.layers):
        # Check if layer types are the same
        if type(layer1) != type(layer2):
            return False
        
        # Check if layer configurations are the same
        if layer1.get_config() != layer2.get_config():
            return False
        
        # Check if weights are the same
        weights1 = layer1.get_weights()
        weights2 = layer2.get_weights()
        
        if len(weights1) != len(weights2):
            return False
        
        for w1, w2 in zip(weights1, weights2):
            if not np.allclose(w1, w2, atol=1e-5):
                return False
    
    return True

def try_load_model(model_path: Path):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        custom_objects = {
        }
        return keras.models.load_model(model_path, custom_objects=custom_objects)
    except Exception as e:
        traceback.print_exc()
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(f"{exc_type} {exc_obj} {exc_tb.tb_lineno}")
        print(f"Failed to load model from {model_path}")
        return None


class CosineLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, total_batches, initial_lr, min_lr=1e-6):
        super(CosineLearningRateScheduler, self).__init__()
        self.total_batches = total_batches
        self.initial_lr = initial_lr
        self.min_lr = min_lr

    def on_batch_end(self, batch, logs=None):
        # Compute the cosine learning rate
        new_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + math.cos(math.pi * batch / self.total_batches))
        self.model.optimizer.learning_rate.assign(new_lr)
        

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
                break


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    main(args)
