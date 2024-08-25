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


def main(args):
    character_llm_training()

def character_llm_training():
    data_dir = Path(project_config['data_dir'])
    input_length = project_config['llm']['input_length']
    batch_size = project_config['llm']['batch_size']
    character_embedding_dim = project_config['llm']['character_embedding_dim']
    num_epochs = project_config['llm']['epochs']

    dataset_text_file = project_config['llm'].get('dataset_text_file', None)
    dataset_text_file = data_dir / dataset_text_file

    alphabet = get_alphabet(dataset_text_file)
    print_project_config(project_config)

    print(f"Alphabet size: {len(alphabet)}")

    char_to_index = {char: index for index, char in enumerate(sorted(alphabet))}
    index_to_char = {index: char for char, index in char_to_index.items()}

    fresh_model = kermit_language_model(
        input_length=input_length,
        alphabet_size=len(alphabet),
        character_embedding_dim=character_embedding_dim
    )

    model_path = data_dir / "models" / "imdb_autoencoder.keras"
    # loaded_model = try_load_model(model_path)
    loaded_model = None

    if loaded_model is None:
        print("\n> No model found, using new model\n")
        model = fresh_model
    elif not are_models_same(fresh_model, loaded_model):
        print("\n> Loaded model is out of date, using new model\n")
        model = fresh_model
    else:
        print("\n> Using loaded model\n")
        model = loaded_model

    print("\n> compiling model\n")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=linear_mask_loss,
        metrics=[NextTokenAccuracy()],
    )

    model.summary()

    dataset = load_text_file_dataset(dataset_text_file)
    tf_dataset = create_tf_dataset(dataset, char_to_index, input_length, batch_size)

    save_model_callback = keras.callbacks.ModelCheckpoint(
        filepath=str(model_path),
        save_best_only=False,
        monitor="loss",
        verbose=1,
        save_freq=1000
    )

    reconstruct_callback = ReconstructCallback(
        char_to_index=char_to_index,
        index_to_char=index_to_char,
        max_input_length=input_length,
        num_samples=5,
        frequency=1000,
        gen_length=200,
        text_file=dataset_text_file
    )

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

    steps_per_epoch = calculate_steps_per_epoch(dataset_text_file, input_length, batch_size)

    starting_lr = 0.0002
    cosine_lr_scheduler = CosineLearningRateScheduler(
        total_batches=steps_per_epoch,
        initial_lr=starting_lr,
        min_lr=starting_lr
    )

    model.fit(
        tf_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        callbacks=[
            learning_rate_callback,
            save_model_callback,
            commit_changes_to_git_callback,
            tensorboard_callback,
            reconstruct_callback,
            keras.callbacks.TensorBoard('./logs', update_freq=5)
        ]
    )


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

class TokenProcessor:
    def __init__(self, char_to_index, max_input_length):
        self.char_to_index = char_to_index
        self.index_to_char = {index: char for char, index in char_to_index.items()}
        self.max_input_length = max_input_length

    def tokenize(self, input_string, truncate=False):
        input_indices = [self.char_to_index.get(char, self.char_to_index[SPECIAL_PAD]) for char in input_string]
        
        if truncate:
            # Truncate or pad the sequence to max_input_length
            if len(input_indices) > self.max_input_length:
                input_indices = input_indices[:self.max_input_length]
            elif len(input_indices) < self.max_input_length:
                input_indices += [self.char_to_index[SPECIAL_PAD]] * (self.max_input_length - len(input_indices))
        else:
            # Only pad if necessary, don't truncate
            if len(input_indices) < self.max_input_length:
                input_indices += [self.char_to_index[SPECIAL_PAD]] * (self.max_input_length - len(input_indices))
        
        return np.array(input_indices)
    

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


def load_text_file_dataset(file_path, chunk_size=1024 * 1024):  # 1MB chunks
    def generate_chunks():
        with open(file_path, 'r', encoding='utf-8') as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    return tf.data.Dataset.from_generator(
        generate_chunks,
        output_types=tf.string,
        output_shapes=()
    )

def preprocess_text(text):
    text = remove_excessive_whitespace(text)
    return text


def remove_excessive_whitespace(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def create_tf_dataset(dataset, char_to_index, max_input_length, batch_size):
    token_processor = TokenProcessor(char_to_index, max_input_length)

    def generate_samples(text_chunk):
        text = preprocess_text(text_chunk.numpy().decode('utf-8'))
        tokenized = token_processor.tokenize(text)
        for i in range(len(tokenized) - max_input_length):
            input_sequence = tokenized[i:i+max_input_length]
            target_sequence = tokenized[i+1:i+max_input_length+1]
            if len(target_sequence) < max_input_length:
                target_sequence = np.pad(target_sequence, (0, max_input_length - len(target_sequence)), 'constant', constant_values=char_to_index[SPECIAL_PAD])
            yield input_sequence, target_sequence

    def generator():
        for chunk in dataset:
            yield from generate_samples(chunk)

    return tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.int32, tf.int32),
        output_shapes=((max_input_length,), (max_input_length,))
    ).map(lambda x, y: (x, tf.one_hot(y, depth=len(char_to_index)))
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)


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
    last_token = Reshape((1, input_sequence.shape[-1]), name=n("reshape_last_token"))(input_sequence[:, -1, :])

    input_length = input_sequence.shape[1]

    last_token_positional_mask = Dense(input_length, activation=None, name=n("dense_last_token_positional_mask"))(last_token)
    last_token_positional_mask = Reshape((input_length, 1), name=n("reshape_last_token_positional_mask_2"))(last_token_positional_mask)

    position_encoded = input_sequence + positional_encoding
    tokenwise_positional_mask = moe_1x1_projection(input_dim=token_dim, output_dim=1, num_experts=1, activation=None)(position_encoded)

    positional_mask = Softmax(axis=1, name=n("softmax_positional_mask"))(tokenwise_positional_mask)

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


def moe_1x1_projection(input_dim, output_dim, num_experts, activation=None, mask=False):
    # Stack the expert weights into a single tensor of shape (num_experts, D, D)
    expert_weights = [tf.Variable(tf.random.normal([input_dim, output_dim]), name=n(f"expert_{i}_weight")) for i in range(num_experts)]
    expert_weights = Stack(axis=0, name=n("stack_expert_weights"))(expert_weights)
    
    # Expand dimensions of expert_weights for broadcasting
    expert_weights = Lambda(lambda x: tf.expand_dims(x, axis=0), name=n("expand_expert_weights"))(expert_weights)
    
    def apply_moe_layer(input_sequence):
        input_length = input_sequence.shape[1]
        # Project each token to E scores using a Dense layer

        if mask:
            input_mask = Dense(1, name=n("input_mask_projection"), use_bias=False, activation=None)(input_sequence)
            input_mask = Softmax(name=n("input_mask_softmax"), axis=1)(input_mask)
            masked_input = Multiply(name=n("multiply_mask"))([input_sequence, input_mask])
            expert_scores = Conv1D(filters=num_experts, kernel_size=1, activation=None, name=n("expert_score_projection"))(masked_input)
        else:
            expert_scores = Conv1D(filters=num_experts, kernel_size=1, activation=None, name=n("expert_score_projection"))(input_sequence)

        
        # Sum along the channel dimension to get E scores
        expert_scores_summed = Lambda(lambda x: tf.reduce_sum(x, axis=1), name=n("sum_scores"))(expert_scores) / sqrt(input_length)
        
        # Softmax the scores
        expert_scores_softmax = Softmax(name=n("expert_score_softmax"))(expert_scores_summed)
        
        # Reshape the softmax scores to match the shape needed for multiplication (batch_size, num_experts, 1)
        expert_scores_softmax = Reshape((num_experts, 1, 1), name=n("reshape_scores"))(expert_scores_softmax)

        tf.print(f"shapes of all the stuff: expert_scores: {expert_scores.shape}, expert_scores_summed: {expert_scores_summed.shape}, expert_scores_softmax: {expert_scores_softmax.shape}, expert_weights: {expert_weights.shape}")
        
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

    # Expand dimensions of expert_weights for broadcasting
    expert_weights = Lambda(lambda x: tf.expand_dims(x, axis=0), name=n("expand_expert_weights"))(expert_weights)
    
    def generate_weights(input_vec):
        # Project the input vector to a set of scores for the experts
        expert_scores = Dense(num_experts, activation=None, name=n("expert_score_projection"))(input_vec)
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

######################################################################################################################################
#-------------------------------------------------------------------------------------------------------------------------------------
#.....................................................................................................................................

def kermit_language_model(input_length, alphabet_size, character_embedding_dim, num_blocks=4, num_heads=4):
    activation = 'leaky_relu'

    inputs = tf.keras.Input(shape=(input_length,), dtype=tf.int32, name=n("input_layer"))
    embedding_layer = Embedding(input_dim=alphabet_size, output_dim=character_embedding_dim, name=n("embedding_layer"))
    embedding = embedding_layer(inputs)
    
    head_dim = character_embedding_dim // num_heads

    full_dim_experts = create_experts(character_embedding_dim, character_embedding_dim, num_experts=2)

    
    # Downsample the sequence
    x = Conv1D(filters=character_embedding_dim, kernel_size=3, strides=2, padding='same', activation=None, name=n("conv_downsample_1"))(embedding)
    x = Conv1D(filters=character_embedding_dim, kernel_size=3, strides=2, padding='same', activation=None, name=n("conv_downsample_2"))(x)
    x = LayerNormalization(name=n("layer_norm_downsample"))(x)

    positional_encoding = sinusoidal_encoding(input_length//4, character_embedding_dim//num_heads)

    for i in range(num_blocks):
        # Apply self-attention and feed-forward layers to the entire sequence
        head_dim_experts = create_experts(head_dim, head_dim, num_experts=16)
        x = additive_self_attention(character_embedding_dim, num_heads=num_heads, experts=head_dim_experts, positional_encoding=positional_encoding)(x)
        x = ff_layer(character_embedding_dim, activation)(x)
 
    # Up-convolution: Restore the sequence length to the original input_length
    x = Conv1DTranspose(filters=character_embedding_dim, kernel_size=3, strides=2, padding='same', name=n("conv_upsample_1"))(x)
    x = Conv1DTranspose(filters=character_embedding_dim, kernel_size=3, strides=2, padding='same', name=n("conv_upsample_2"))(x)
    x = LayerNormalization(name=n("layer_norm_upsample"))(x)
    embedding_residual = Conv1D(filters=character_embedding_dim, kernel_size=5, padding='same', activation=None, name=n("conv_embedding_residual"))(embedding)
    x = Add(name=n("add_embedding_residual"))([x, embedding_residual])
    # Apply the embedding layer to convert the output to logits over the vocabulary
    x = Lambda(lambda x: tf.matmul(x, embedding_layer.embeddings, transpose_b=True), name=n("lambda_final_matmul"), output_shape=(input_length, alphabet_size))(x)

    # Apply softmax to get the probability distribution over the vocabulary for each token in the sequence
    x = Softmax(name=n("softmax_output"))(x)

    model = tf.keras.Model(inputs=inputs, outputs=x, name="KermitLanguageModel")
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
    def __init__(self, char_to_index, index_to_char, max_input_length, num_samples, frequency, text_file, temperature=1.0, gen_length=100):
        self.char_to_index = char_to_index
        self.index_to_char = index_to_char
        self.max_input_length = max_input_length
        self.num_samples = num_samples
        self.frequency = frequency
        self.dataset = load_text_file_dataset(text_file)
        self.temperature = temperature
        self.gen_length = gen_length

    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch + 1} Autoencoder Outputs:\n")
        self._print_predictions()

    def on_batch_end(self, batch, logs=None):
        if batch % self.frequency == 0:
            print(f"\nBatch {batch} Autoencoder Outputs:\n")
            self._print_predictions()

    def _get_random_sequence(self):
        chunk = next(iter(self.dataset.take(1)))
        text = chunk.numpy().decode('utf-8')
        if len(text) > self.max_input_length:
            start = np.random.randint(0, len(text) - self.max_input_length)
            return text[start:start+self.max_input_length]
        return text.ljust(self.max_input_length)  # Pad if shorter than max_input_length
    
    def _tokenize(self, sequence):
        return np.array([self.char_to_index.get(char, self.char_to_index[SPECIAL_PAD]) for char in sequence], dtype=np.int32)

    def _select_top_character(self, predictions):
        return np.argmax(predictions, axis=-1)

    def _print_predictions(self):
        for _ in range(self.num_samples):
            input_sequence = self._get_random_sequence()
            input_tensor = self._tokenize(input_sequence)
            
            generated_sequence = list(input_sequence)
            color_coded_sequence = []
            
            for i in range(self.gen_length):
                try:
                    input_tensor_reshaped = np.reshape(input_tensor, (1, self.max_input_length))
                    predictions = self.model.predict(input_tensor_reshaped, verbose=0)[0]

                    # Focus on the prediction for the last token in the sequence
                    last_token_prediction = predictions[-1]

                    if i == 0:
                        top_5_indices = np.argsort(last_token_prediction)[-5:][::-1]
                        top_5_chars = [self.index_to_char[idx] for idx in top_5_indices]
                        top_5_probs = last_token_prediction[top_5_indices]
                        print(f"Step {i} - Top 5 predictions for the last token:")
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
                    
                except Exception as e:
                    print(f"Error during prediction: {e}")
                    break
            
            print(f"Prediction | {''.join(generated_sequence[:self.max_input_length])}<|>{''.join(color_coded_sequence)} |")
            print("\n")

    def _color_code(self, char, confidence):
        if confidence > 0.92:
            return f"\033[30;42m{char}\033[0m"  # Green background, black text
        elif confidence > 0.7:
            return f"\033[30;43m{char}\033[0m"  # Yellow background, black text
        else:
            return f"\033[30;41m{char}\033[0m"  # Red background, black text




def calculate_steps_per_epoch(file_path: Path, max_input_length, batch_size, cache_file='dataset_size_cache.json'):
    cache_key = f"file_{hashlib.md5(str(file_path).encode()).hexdigest()}"
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
            if cache_key in cached_data:
                total_windows = cached_data[cache_key]
                print(f"Using cached total windows: {total_windows}")
                return total_windows // batch_size

    dataset = load_text_file_dataset(file_path)

    total_windows = 0

    for chunk in tqdm(dataset, desc="Calculating windows", unit="chunk"):
        text_length = len(chunk.numpy().decode('utf-8'))
        # Calculate the number of windows in this chunk
        chunk_windows = max(0, (text_length - max_input_length) + 1)
        total_windows += chunk_windows

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

    print(f"Total windows across all chunks: {total_windows}")
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {steps_per_epoch}")

    return steps_per_epoch


def get_alphabet(file_path: Path, cache_file: str = 'overall_unique_chars.json') -> Set[str]:
    cache_key = f"file_{hashlib.md5(str(file_path).encode()).hexdigest()}"
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
            if cache_key in cached_data:
                alphabet = set(cached_data[cache_key])
                print(f"Using cached alphabet for {cache_key}")
                alphabet.update([SPECIAL_PAD, SPECIAL_START, SPECIAL_END])
                return alphabet

    dataset = load_text_file_dataset(file_path)
    
    alphabet = set()
    for chunk in tqdm(dataset, desc="Processing dataset", unit="chunk"):
        alphabet.update(set(chunk.numpy().decode('utf-8')))

    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
    else:
        cached_data = {}
    
    cached_data[cache_key] = list(alphabet)
    with open(cache_file, 'w') as f:
        json.dump(cached_data, f)

    # Add special tokens to the alphabet
    alphabet.update([SPECIAL_PAD, SPECIAL_START, SPECIAL_END])
    return alphabet


def gelu_activation(x):
    return tf.keras.activations.gelu(x, approximate=True)


class SinActivation(Layer):
    def call(self, inputs):
        return tf.math.sin(inputs)


class Sigmoid(Layer):
    def call(self, inputs):
        return tf.math.sigmoid(inputs)


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
