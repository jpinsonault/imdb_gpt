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
    def __init__(self, char_to_index, max_input_length, mask_percentage):
        self.char_to_index = char_to_index
        self.index_to_char = {index: char for char,
                              index in char_to_index.items()}
        self.index_to_char[char_to_index[SPECIAL_PAD]] = '@'
        self.max_input_length = max_input_length
        self.mask_percentage = mask_percentage

    def tokenize(self, input_string):
        if len(input_string) > self.max_input_length:
            print(f"Warning: Input string is longer than max input length of {
                  self.max_input_length}: {input_string}")
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
    # data_dir                = Path(project_config['docker_data_dir_mount'])
    data_dir = Path(project_config['data_dir'])
    db_path = data_dir / 'imdb.db'
    max_input_length = project_config['entities']['max_entity_length']
    batch_size = project_config['search_autoencoder']['batch_size']
    character_embedding_dim = project_config['search_autoencoder']['character_embedding_dim']

    alphabet = get_alphabet(db_path)

    print_project_config(project_config)

    alphabet = [SPECIAL_PAD] + alphabet

    char_to_index = {char: index for index, char in enumerate(alphabet)}
    index_to_char = {index: char for index, char in enumerate(alphabet)}

    num_blocks_per_resolution = 1

    fresh_model = kermit_autoencoder(input_length=max_input_length,
                                     alphabet_size=len(alphabet),
                                     character_embedding_dim=character_embedding_dim,
                                     num_blocks_per_resolution=num_blocks_per_resolution)

    model_path = data_dir / "models" / "imdb_autoencoder.keras"
    loaded_model = try_load_model(model_path)

    # compare the two models and if there's a difference, use the new model because
    # the loaded model is out of date
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
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=MaskedCategoricalCrossentropy(char_to_index=char_to_index),
        metrics=["accuracy"],
    )

    model.summary()

    masked_percentage = 0.15
    token_processor = TokenProcessor(
        char_to_index=char_to_index,
        max_input_length=max_input_length,
        mask_percentage=masked_percentage,
    )

    dataset = tf.data.Dataset.from_generator(lambda: autoencoder_batch_generator(db_path, batch_size, token_processor),
                                             output_types=(
                                                 tf.int32, tf.float32),
                                             output_shapes=((batch_size, max_input_length, ),
                                                            (batch_size, max_input_length, len(char_to_index))))

    save_frequency = 50

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
        db_path, token_processor, num_samples=10, frequency=save_frequency)

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


class ContinuousPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, P):
        super(ContinuousPositionalEncoding, self).__init__()
        self.P = P

    def call(self, x):
        position = tf.expand_dims(x, -1)  # shape: (batch_size, 1)
        div_term = tf.exp(tf.range(0, self.P, 2, dtype=tf.float32)
                          * -(np.log(10000.0) / self.P))

        sin_terms = tf.sin(position * div_term)
        cos_terms = tf.cos(position * div_term)

        # Concatenate sin and cos terms along the last dimension
        encoding = tf.concat([sin_terms, cos_terms], axis=-1)

        return Flatten()(encoding)


class FoveatedAttention(Layer):
    """
    Foveated Attention Layer for Neural Networks.
    """

    def __init__(self, num_foveas, character_embedding_dim, fovea_embedding_dim, fixed_width, return_context=False, **kwargs):
        super(FoveatedAttention, self).__init__(**kwargs)
        self.sin_activation = SinActivation()

        self.num_foveas = num_foveas
        self.fixed_width = fixed_width
        self.return_context = return_context
        self.character_embedding_dim = character_embedding_dim
        self.fovea_embedding_dim = fovea_embedding_dim

        self.center_prediction_1 = [Dense(fovea_embedding_dim, activation=self.sin_activation)
                                    for _ in range(num_foveas)]
        self.center_prediction_2 = [Dense(fovea_embedding_dim, activation=self.sin_activation)
                                    for _ in range(num_foveas)]
        self.center_prediction_3 = [Dense(fovea_embedding_dim, activation=self.sin_activation)
                                    for _ in range(num_foveas)]
        self.center_prediction_end = [Dense(1, activation=None)
                              for _ in range(num_foveas)]
        
        self.center_prediction_layer_norms = [LayerNormalization() for _ in range(num_foveas)]

        self.context_dense_layers_start = [Dense(fovea_embedding_dim, activation=self.sin_activation)
                                            for _ in range(num_foveas)]
        self.context_dense_layers_end = [Dense(fovea_embedding_dim, activation=self.sin_activation)
                                     for _ in range(num_foveas)]
        self.context_dense_layer_norms = [LayerNormalization() for _ in range(num_foveas)]

        self.compressed_context_bottleneck = Dense(character_embedding_dim, activation=self.sin_activation)
        self.compressed_context_embiggen = Dense(fovea_embedding_dim, activation=self.sin_activation)
        self.compressed_context_layer_norm = LayerNormalization()

        self.at_full = Conv1D(filters=character_embedding_dim, kernel_size=5,
                              strides=1, padding="same", activation=self.sin_activation)
        self.to_one_4th = Conv1D(filters=character_embedding_dim*2, kernel_size=8,
                                 strides=4, padding="same", activation=self.sin_activation)
        self.at_one_4th = Conv1D(filters=character_embedding_dim*2, kernel_size=4,
                                 strides=1, padding="same", activation=self.sin_activation)
        self.to_one_16th = Conv1D(filters=character_embedding_dim*4, kernel_size=8,
                                  strides=4, padding="same", activation=self.sin_activation)
        self.at_one_16th = Conv1D(filters=character_embedding_dim*4, kernel_size=4,
                                  strides=1, padding="same", activation=self.sin_activation)

        self.positional_encoding = ContinuousPositionalEncoding(
            P=character_embedding_dim)

        self.layer_norm_at_full = LayerNormalization()
        self.layer_norm_at_one_4th = LayerNormalization()
        self.layer_norm_at_one_16th = LayerNormalization()

    def build(self, input_shape):
        self.channels = input_shape[-1]
        super(FoveatedAttention, self).build(input_shape)

    def call(self, input_list) -> Tuple[Layer, Layer, Layer]:
        token_inputs = input_list[0]
        batch_size = tf.shape(token_inputs)[0]
        should_produce_context = len(input_list) == 1
        previous_fovea_outputs = input_list[1] if len(input_list) > 1 else 0
        previous_fovea_centers = input_list[2] if len(input_list) > 1 else [tf.zeros((batch_size, 1)) for _ in range(self.num_foveas)]

        length = tf.shape(token_inputs)[1]

        if should_produce_context:
            global_context = self.convolute_inputs_from_full_to_one_16th(token_inputs)
            compressed_context = self.compressed_context_bottleneck(global_context)
            compressed_context = self.compressed_context_embiggen(compressed_context)
            compressed_context = self.compressed_context_layer_norm(compressed_context)
        else:
            global_context = previous_fovea_outputs
            compressed_context = 0

        foveated_outputs, fovea_centers = self.compute_foveas(token_inputs, previous_fovea_centers, global_context, batch_size, length)

        final_representation = compressed_context + (Add()(foveated_outputs) / tf.sqrt(tf.cast(self.num_foveas, tf.float32)))

        return final_representation, fovea_centers

    def compute_foveas(self, token_inputs, previous_fovea_centers, global_context, batch_size, length):
        foveated_outputs = []
        fovea_centers = []

        positional_encodings = [self.positional_encoding(center) for center in previous_fovea_centers]
        positional_encodings = tf.concat(positional_encodings, axis=-1)

        for i in range(self.num_foveas):
            previous_fovea_center = previous_fovea_centers[i]

            positional_encoding = self.positional_encoding(previous_fovea_center)

            context_and_position = Concatenate()([global_context, positional_encoding])

            fovea_center = self.center_prediction_1[i](context_and_position)
            fovea_center = self.center_prediction_2[i](fovea_center)
            fovea_center = self.center_prediction_3[i](fovea_center)
            fovea_center = self.center_prediction_layer_norms[i](fovea_center)

            fovea_center = self.center_prediction_end[i](fovea_center)

            fovea_center = Sigmoid()(fovea_center + previous_fovea_center)

            fovea_centers.append(fovea_center)

            positional_encoding = self.positional_encoding(fovea_center)

            center_position = tf.cast(
                fovea_center * tf.cast(length, tf.float32), tf.int32)

            # Generate masks for each fovea
            row_indices = tf.range(length)
            row_indices = tf.expand_dims(row_indices, 0)
            row_indices = tf.tile(row_indices, [batch_size, 1])

            masks = tf.logical_and(
                row_indices >= center_position - self.fixed_width // 2,
                row_indices < center_position + self.fixed_width // 2
            )
            masks = tf.cast(masks, tf.float32)

            # Apply masks to extract foveas
            masks = tf.expand_dims(masks, -1)
            foveas = token_inputs * masks

            # Ensure no zero-length segments
            mask_sums = tf.reduce_sum(masks, axis=1)
            # Replace zero-length segments with 1 to avoid NaNs
            mask_sums = tf.where(mask_sums == 0, 1.0, mask_sums)

            fovea_means = tf.reduce_sum(foveas, axis=1) / mask_sums

            means_and_positions = Concatenate()(
                [fovea_means, positional_encoding])

            # Process through dense layer
            fovea_features = self.context_dense_layers_end[i](means_and_positions)
            fovea_features = self.context_dense_layer_norms[i](fovea_features)

            foveated_outputs.append(fovea_features)

        return foveated_outputs, fovea_centers

    def convolute_inputs_from_full_to_one_16th(self, inputs):
        full = self.at_full(inputs)
        full = Add()([full, inputs])
        full = self.layer_norm_at_full(full)
        to_one_fourth = self.to_one_4th(full)
        one_fourth = self.at_one_4th(to_one_fourth)
        one_fourth = Add()([one_fourth, to_one_fourth])
        one_fourth = self.layer_norm_at_one_4th(one_fourth)
        to_one_16th = self.to_one_16th(one_fourth)
        one_16th = self.at_one_16th(to_one_16th)
        one_16th = Add()([one_16th, to_one_16th])
        one_16th = self.layer_norm_at_one_16th(one_16th)
        return Flatten()(one_16th)

    def get_config(self):
        config = super(FoveatedAttention, self).get_config()
        config.update({
            'num_foveas': self.num_foveas,
            'embedding_size': self.context_dense_layers_end[0].units,
            'fixed_width': self.fixed_width,
        })
        return config


class StatsLayer(Layer):
    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=1, keepdims=True)
        stddev = tf.math.reduce_std(inputs, axis=1, keepdims=True)
        min_val = tf.reduce_min(inputs, axis=1, keepdims=True)
        max_val = tf.reduce_max(inputs, axis=1, keepdims=True)
        return Concatenate()([mean, stddev, min_val, max_val])


def entropy_boosted_residual_layer(use_entropy_boost=True):
    """
    A layer that applies a residual connection and optionally scales its output based on the entropy of the input.
    
    Args:
    use_entropy_boost: Boolean, if True, apply entropy-based scaling. If False, just add the residual.
    
    Returns:
    A Keras layer that can be used in a model
    """
    def calculate_entropy(tensor):
        normalized = tf.nn.softmax(tensor, axis=-1)
        entropy = -tf.reduce_sum(normalized * tf.math.log(normalized + 1e-10), axis=-1)
        max_entropy = tf.math.log(tf.cast(tf.shape(normalized)[-1], tf.float32))
        normalized_entropy = entropy / max_entropy
        return tf.reduce_mean(normalized_entropy)

    def call(inputs):
        x, residual = inputs
        
        if use_entropy_boost:
            input_entropy = calculate_entropy(x)
            scale = tf.where(input_entropy > 0.5, 
                             1.0 + (input_entropy - 0.5), 
                             1.0 - (0.5 - input_entropy))
            scaled_residual = scale * residual
            return x + scaled_residual
        else:
            return x + residual

    return tf.keras.layers.Lambda(call)

def residual_conv1d_block(x, filters, kernel_size, activation, num_blocks, name, strides=1):
    for i in range(num_blocks):
        residual = x
        starting_conv = DepthwiseConv1D(kernel_size=kernel_size, strides=strides, padding="same", activation=activation, name=f"{name}_starting_conv_{i}")
        reduction = Conv1D(filters=filters//4, kernel_size=1, strides=1, padding="same", activation=activation, name=f"{name}_reduction_{i}")
        expansion = Conv1D(filters=filters, kernel_size=1, strides=1, padding="same", activation=activation, name=f"{name}_expansion_{i}")
        ending_conv = DepthwiseConv1D(kernel_size=kernel_size, strides=strides, padding="same", activation=activation, name=f"{name}_ending_conv_{i}")

        x = starting_conv(x)
        x = reduction(x)
        x = expansion(x)
        x = ending_conv(x)

        x = x + residual
        x = LayerNormalization()(x)
    return x


def entropy_minimization_layer():
    def call(x):
        # Calculate entropy with improved numerical stability
        epsilon = 1e-7
        x_abs = tf.abs(x)
        x_norm = x_abs / (tf.reduce_sum(x_abs, axis=-1, keepdims=True) + epsilon)
        entropy = -tf.reduce_sum(x_norm * tf.math.log(x_norm + epsilon), axis=-1)
        
        # Normalize entropy to [0, 1]
        max_entropy = tf.math.log(tf.cast(tf.shape(x)[-1], tf.float32))
        normalized_entropy = entropy / (max_entropy + epsilon)
        
        # Clip normalized entropy to avoid extreme values
        normalized_entropy = tf.clip_by_value(normalized_entropy, 0.0, 1.0)
        
        # Scale the input based on entropy, preserving sign
        scale = 1 - tf.expand_dims(normalized_entropy, -1)
        result = x * scale
        
        return result
    
    return Lambda(call)

def kermit_autoencoder(input_length, alphabet_size, character_embedding_dim, num_blocks_per_resolution):
    default_activation = 'gelu'

    inputs = Input(shape=(input_length,), dtype=tf.int32)
    embedding = Embedding(input_dim=alphabet_size,
                          output_dim=character_embedding_dim, name="embedding")(inputs)

    conv_down_to_half = Conv1D(filters=character_embedding_dim*2, kernel_size=5, strides=2, padding="same", activation=default_activation)(embedding)
    conv_down_to_half = LayerNormalization()(conv_down_to_half)

    # Apply entropy_boosted_residual_layer here
    residual_half = residual_conv1d_block(conv_down_to_half, filters=character_embedding_dim*2, kernel_size=3, activation=default_activation, num_blocks=num_blocks_per_resolution, name="residual_conv1d_block_half")
    x = entropy_boosted_residual_layer(use_entropy_boost=False)([conv_down_to_half, residual_half])

    conv_down_to_quarter = Conv1D(filters=character_embedding_dim*4, kernel_size=5, strides=2, padding="same", activation=default_activation)(x)
    conv_down_to_quarter = LayerNormalization()(conv_down_to_quarter)

    # Apply entropy_boosted_residual_layer here as well
    residual_quarter = residual_conv1d_block(conv_down_to_quarter, filters=character_embedding_dim*4, kernel_size=3, activation=default_activation, num_blocks=num_blocks_per_resolution, name="residual_conv1d_block_quarter")
    x = entropy_boosted_residual_layer(use_entropy_boost=False)([conv_down_to_quarter, residual_quarter])

    shrink_channels = Conv1D(filters=character_embedding_dim, kernel_size=1, strides=1, padding="same", activation=default_activation)(x)
    shrink_channels = LayerNormalization()(shrink_channels)

    condensed_convolutions = Dense(character_embedding_dim*2, activation=default_activation)(Flatten()(x))
    
    # Apply entropy_boosted_residual_layer to the dense layers
    dense_residual = Dense(character_embedding_dim*2, activation=default_activation)(
        Dense(character_embedding_dim*2, activation=default_activation)(condensed_convolutions)
    )
    condensed_convolutions = entropy_boosted_residual_layer(use_entropy_boost=False)([condensed_convolutions, dense_residual])

    final_representation = Dense(input_length*4, activation=default_activation)(Flatten()(embedding))
    final_representation = Dense(input_length, activation=default_activation)(final_representation)
    final_representation = LayerNormalization()(final_representation)

####################################################################################################
    bottleneck = entropy_minimization_layer()(final_representation)
####################################################################################################d

    x = Dense(input_length*character_embedding_dim, activation=default_activation)(Flatten()(bottleneck))
    x = Dense(input_length*character_embedding_dim, activation=default_activation)(Flatten()(x))
    x = LayerNormalization()(x)
    x = Reshape((input_length, character_embedding_dim))(x)

    x = Conv1D(filters=alphabet_size, kernel_size=1, strides=1, padding="same", activation=default_activation)(x)
    
    # Apply entropy_boosted_residual_layer to the final convolutional layers
    conv_residual = Conv1D(filters=alphabet_size, kernel_size=1, strides=1, padding="same", activation=default_activation)(
        Conv1D(filters=alphabet_size//2, kernel_size=1, strides=1, padding="same", activation=default_activation)(x)
    )
    x = entropy_boosted_residual_layer(use_entropy_boost=False)([x, conv_residual])
    
    x = LayerNormalization()(x)
    x = Conv1D(filters=alphabet_size, kernel_size=1, strides=1, padding="same", activation="softmax")(x)

    model = Model(inputs=inputs, outputs=x, name="kermit_autoencoder")
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

def autoencoder_batch_generator(db_path, batch_size, token_processor: TokenProcessor, SPECIAL_PAD='PAD'):
    while True:
        # Set up batches
        x_batch = np.zeros(
            (batch_size, token_processor.max_input_length), dtype=np.int32)
        y_batch = np.zeros((batch_size, token_processor.max_input_length, len(
            token_processor.char_to_index)), dtype=np.float32)

        # Database connection
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Initialize batch index
        batch_index = 0

        query = "SELECT entityName FROM entity_vectors WHERE LENGTH(entityName) < ?"
        cursor.execute(query, (token_processor.max_input_length,))

        all_entities = cursor.fetchall()

        conn.close()

        print("Entity Batch Generator: Fetching entity names from entity_vectors table...")
        for entity_name in all_entities:
            entity_name = entity_name[0].strip()

            if len(entity_name) > token_processor.max_input_length or len(entity_name) == 0:
                print(f"Skipping entity name: {entity_name}")
                continue

            try:
                input_string_token_indices = token_processor.tokenize(
                    entity_name)

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
    def __init__(self, db_path, token_processor: TokenProcessor, num_samples, frequency):
        super().__init__()
        self.db_path = db_path
        self.token_processor = token_processor
        self.num_samples = num_samples
        self.frequency = frequency

    def on_batch_end(self, epoch, logs=None):
        if epoch % self.frequency != 0:
            return

        # Database connection
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Fetch a random sample of entity names less than max_input_length
        cursor.execute(f"SELECT entityName FROM entity_vectors WHERE LENGTH(entityName) < {
                       self.token_processor.max_input_length} ORDER BY RANDOM() LIMIT {self.num_samples}")
        selected_rows = cursor.fetchall()

        conn.close()

        print(f"\nReconstruction examples at the end of epoch {epoch}:")

        for row in selected_rows:
            original = row[0].strip()
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


def get_alphabet(db_path):
    # Connect to the SQLite database
    try:
        conn = sqlite3.connect(db_path)
    except Exception as e:
        print(f"Error connecting to database file: {db_path}: {e}")
        exit(1)
    cursor = conn.cursor()

    # Query the alphabet table
    cursor.execute("SELECT characters FROM alphabet")

    # Fetch the result
    result = cursor.fetchone()

    # Close the database connection
    conn.close()

    if result:
        # Parse the JSON field to get the list of characters

        alphabet = json.loads(result[0])
        alphabet = [chr(ord(c)) for c in alphabet]
        return alphabet
    else:
        # Return an empty list if no data is found
        return []


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    main(args)
