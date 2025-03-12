from functools import partial
from prettytable import PrettyTable
import numpy as np
import tensorflow as tf
from enum import Enum
from tensorflow.keras.layers import (
    Dense,
    Embedding,
    Input,
    LayerNormalization,
    Reshape,
    Softmax,
    Dropout,
    Add,
    Lambda,
    Flatten,
    Conv1D,
    Conv1DTranspose,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
)
from tensorflow.keras.models import Model

import abc
from tensorflow.keras.layers import Layer, Subtract

################################################################################
# Enums and constants
################################################################################

class Scaling(Enum):
    NONE = 1
    NORMALIZE = 2
    STANDARDIZE = 3
    LOG = 4

    def none_transform(self, x, **kwargs):
        return x

    def none_untransform(self, x, **kwargs):
        return x

    def normalize_transform(self, x, min_val, max_val):
        return (x - min_val) / (max_val - min_val) if max_val > min_val else 0.0

    def normalize_untransform(self, x, min_val, max_val):
        return x * (max_val - min_val) + min_val

    def standardize_transform(self, x, mean_val, std_val):
        return (x - mean_val) / std_val if std_val != 0 else 0.0

    def standardize_untransform(self, x, mean_val, std_val):
        return x * std_val + mean_val

    def log_transform(self, x):
        return np.log1p(x)

    def log_untransform(self, x):
        return np.expm1(x)

# Special characters for text fields
SPECIAL_PAD = '\u200C'
SPECIAL_START = '\u200D'
SPECIAL_END = '\u200E'
SPECIAL_SEP = '\u200F'
SPECIAL_PAD_DISPLAY = '_'
SPECIAL_START_DISPLAY = '<|'
SPECIAL_END_DISPLAY = '|>'
SPECIAL_SEP_DISPLAY = '<|>'

################################################################################
# Utility for text fields
################################################################################
@tf.keras.utils.register_keras_serializable()
def add_positional_encoding(input_sequence):
    seq_len = tf.shape(input_sequence)[1]
    model_dim = tf.shape(input_sequence)[2]

    positions = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)
    dims = tf.range(model_dim)
    angle_rates = 1 / tf.pow(10000.0, (2 * tf.cast(tf.math.floordiv(dims, 2), tf.float32)) / tf.cast(model_dim, tf.float32))
    angles = positions * angle_rates

    pos_encoding = tf.where(tf.math.floormod(dims, 2) == 0, tf.sin(angles), tf.cos(angles))
    pos_encoding = pos_encoding[tf.newaxis, ...]  # Expand batch dimension

    return input_sequence + pos_encoding


@tf.keras.utils.register_keras_serializable()
def sin_activation(x):
    return tf.sin(x)

################################################################################
# BaseField
################################################################################

class BaseField(abc.ABC):
    def __init__(self, name: str, optional: bool = False):
        self.name = name
        self.optional = optional

    @abc.abstractmethod
    def _get_input_shape(self):
        pass

    @abc.abstractmethod
    def _get_output_shape(self):
        pass

    @abc.abstractmethod
    def _get_loss(self):
        pass

    @abc.abstractmethod
    def _accumulate_stats(self, raw_value):
        pass

    @abc.abstractmethod
    def _finalize_stats(self):
        pass

    @abc.abstractmethod
    def _transform(self, raw_value):
        pass

    @abc.abstractmethod
    def build_encoder(self, latent_dim: int) -> tf.keras.Model:
        pass

    @abc.abstractmethod
    def build_decoder(self, latent_dim: int) -> tf.keras.Model:
        pass

    @abc.abstractmethod
    def to_string(self, predicted_array: np.ndarray) -> str:
        pass

    @abc.abstractmethod
    def print_stats(self):
        pass

    @property
    def input_shape(self):
        base_shape = self._get_input_shape()
        if self.optional:
            return (base_shape[0] + 1,)
        return base_shape

    @property
    def input_dtype(self):
        return tf.float32

    @property
    def output_shape(self):
        base_shape = self._get_output_shape()
        if self.optional:
            return (base_shape[0] + 1,)
        return base_shape

    @property
    def output_dtype(self):
        return tf.float32

    @property
    def loss(self):
        return self._get_loss()

    @property
    def weight(self):
        return self._get_weight()

    def accumulate_stats(self, raw_value):
        self._accumulate_stats(raw_value)

    def finalize_stats(self):
        self._finalize_stats()

    def transform(self, raw_value):
        if raw_value is None and not self.optional:
            raise ValueError(f"Received None for non-optional field: {self.name}")
        
        base_output = self._transform(raw_value)

        if self.optional:
            is_null = 1.0 if raw_value is None else 0.0
            return tf.concat(
                [base_output, tf.constant([is_null], dtype=tf.float32)],
                axis=-1
            )
        else:
            return base_output

    def _optional_loss_wrapper(self, base_loss_fn):
        """
        Wraps the given loss function so that when the field is optional,
        the loss on the main output is multiplied by a mask (which is 0 when
        the field is missing) and the flag loss (computed via BCE) is always applied.
        Assumes that the last element in the last dimension is the null flag.
        """
        def loss_fn(y_true, y_pred):
            # Split the tensors along the last dimension.
            # Assume shape (..., base_dim + 1) where the last element is the flag.
            base_true = y_true[..., :-1]
            flag_true = y_true[..., -1]
            base_pred = y_pred[..., :-1]
            flag_pred = y_pred[..., -1]
            # Create a mask: 1 when field is provided (flag == 0) and 0 when missing (flag == 1)
            mask = 1.0 - flag_true
            # Compute the loss on the main output (for instance, using MSE)
            main_loss = base_loss_fn(base_true, base_pred)
            # Compute the loss on the flag (using binary crossentropy)
            flag_loss = tf.keras.losses.binary_crossentropy(flag_true, flag_pred)
            # Return a weighted sum. 
            return mask * main_loss + flag_loss
        return loss_fn

    @property
    def loss(self):
        """
        When the field is optional, wrap the base loss function.
        Otherwise, return the base loss function directly.
        """
        base_loss = self._get_loss()  # this should be a callable loss function
        if self.optional:
            return self._optional_loss_wrapper(base_loss)
        return base_loss

    def _get_weight(self):
        return 1.0

################################################################################
# BooleanField
################################################################################

class BooleanField(BaseField):
    def __init__(self, name: str, use_bce_loss: bool = True, optional: bool = False):
        super().__init__(name, optional)
        self.use_bce_loss = use_bce_loss
        # For stats
        self.count_total = 0
        self.count_ones = 0

    def _get_input_shape(self):
        return (1,)

    def _get_output_shape(self):
        return (1,)

    def _get_loss(self):
        mse_loss = tf.keras.losses.MeanSquaredError()
        binary_cross_entryopy_loss = tf.keras.losses.BinaryCrossentropy()

        return binary_cross_entryopy_loss if self.use_bce_loss else mse_loss

    def _accumulate_stats(self, raw_value):
        if raw_value is not None:
            try:
                val = float(raw_value)
                val = 1.0 if val == 1.0 else 0.0
                self.count_total += 1
                if val == 1.0:
                    self.count_ones += 1
            except ValueError:
                pass

    def _finalize_stats(self):
        # Nothing else needed
        pass

    def _transform(self, raw_value):
        if raw_value is None:
            val = 0.0
        else:
            try:
                val = float(raw_value)
            except ValueError:
                val = 0.0
        val = 1.0 if val == 1.0 else 0.0
        return tf.constant([val], dtype=tf.float32)

    def to_string(self, predicted_array: np.ndarray) -> str:
        if self.optional:
            bool_part = predicted_array[:-1]
            is_null_part = predicted_array[-1]
            if is_null_part > 0.5:
                return "None"
            prob_true = float(bool_part[0])
        else:
            prob_true = float(predicted_array[0])
        return "True" if prob_true >= 0.5 else "False"

    def build_encoder(self, latent_dim: int) -> tf.keras.Model:
        inp = tf.keras.Input(shape=self.input_shape, name=f"{self.name}_input")
        x = Dense(8, activation='gelu')(inp)
        x = LayerNormalization()(x)
        x = Dense(8, activation='gelu')(x)
        out = LayerNormalization()(x)
        return tf.keras.Model(inp, out, name=f"{self.name}_encoder")

    def build_decoder(self, latent_dim: int) -> tf.keras.Model:
        inp = tf.keras.Input(shape=(latent_dim,), name=f"{self.name}_decoder_in")
        
        x = Dense(8, activation='gelu')(inp)
        x = LayerNormalization()(x)
        x = Dense(8, activation='gelu')(x)
        x = LayerNormalization()(x)
        out = Dense(self.output_shape[0], activation='sigmoid', name=f"{self.name}_decoder")(x)
        return tf.keras.Model(inp, out, name=f"{self.name}_decoder")

    def print_stats(self):
        t = PrettyTable(["Boolean Field", self.name])
        t.add_row(["Count (total)", self.count_total])
        t.add_row(["Count(=1)", self.count_ones])
        print(t)

################################################################################
# ScalarField
################################################################################

class ScalarField(BaseField):
    def __init__(
        self,
        name: str,
        scaling: Scaling = Scaling.NONE,
        clip_max=None,
        optional: bool = False,
    ):
        super().__init__(name, optional)
        self.scaling = scaling
        self.clip_max = clip_max
        # For streaming stats
        self.n = 0
        self.sum_ = 0.0
        self.sum_sq = 0.0
        self.min_val = float("inf")
        self.max_val = float("-inf")
        self.mean_val = 0.0
        self.std_val = 1.0
        self.encoding_dim = 32

    def _get_input_shape(self):
        return (1,)

    def _get_output_shape(self):
        return (1,)

    def _get_loss(self):
        return tf.keras.losses.MeanSquaredError()

    def _accumulate_stats(self, raw_value):
        if raw_value is not None:
            try:
                val = float(raw_value)
                self.n += 1
                self.sum_ += val
                self.sum_sq += val * val
                if val < self.min_val:
                    self.min_val = val
                if val > self.max_val:
                    self.max_val = val
            except ValueError:
                pass

    def _finalize_stats(self):
        if self.n > 0:
            self.mean_val = self.sum_ / self.n
            var = (self.sum_sq / self.n) - (self.mean_val**2)
            self.std_val = np.sqrt(var) if var > 1e-12 else 1.0
        else:
            # Fallback
            self.min_val = 0.0
            self.max_val = 0.0
            self.mean_val = 0.0
            self.std_val = 1.0

    def noise_function(self, x):
        gaussian_noise = tf.keras.layers.GaussianNoise(stddev=0.05)(x)
        return x + gaussian_noise

    def _transform(self, raw_value):
        x = 0.0
        if raw_value is not None:
            try:
                x = float(raw_value)
            except ValueError:
                pass
        if self.clip_max is not None:
            x = min(x, self.clip_max)

        if self.scaling == Scaling.NONE:
            x = self.scaling.none_transform(x)
        elif self.scaling == Scaling.NORMALIZE:
            x = self.scaling.normalize_transform(x, min_val=self.min_val, max_val=self.max_val)
        elif self.scaling == Scaling.STANDARDIZE:
            x = self.scaling.standardize_transform(x, mean_val=self.mean_val, std_val=self.std_val)
        elif self.scaling == Scaling.LOG:
            x = self.scaling.log_transform(x)

        return tf.constant([x], dtype=tf.float32)

    def to_string(self, predicted_array: np.ndarray) -> str:
        if self.optional:
            value_part = predicted_array[:-1]
            is_null_part = predicted_array[-1]
            if is_null_part > 0.5:
                return "None"
            float_val = float(value_part[0])
        else:
            float_val = float(predicted_array[0])
        
        if self.scaling == Scaling.NONE:
            float_val = self.scaling.none_untransform(float_val)
        elif self.scaling == Scaling.NORMALIZE:
            float_val = self.scaling.normalize_untransform(float_val, min_val=self.min_val, max_val=self.max_val)
        elif self.scaling == Scaling.STANDARDIZE:
            float_val = self.scaling.standardize_untransform(float_val, mean_val=self.mean_val, std_val=self.std_val)
        elif self.scaling == Scaling.LOG:
            float_val = self.scaling.log_untransform(float_val)
        
        return f"{float_val:.2f}"

    def build_encoder(self, latent_dim: int) -> tf.keras.Model:
        inp = tf.keras.Input(shape=self.input_shape, name=f"{self.name}_input")
        inp = self.noise_function(inp)
        x = Dense(self.encoding_dim, activation='gelu')(inp)
        x = Dense(self.encoding_dim, activation='gelu')(x)
        out = LayerNormalization()(x)
        return tf.keras.Model(inp, out, name=f"{self.name}_encoder")

    def build_decoder(self, latent_dim: int) -> tf.keras.Model:
        inp = tf.keras.Input(shape=(latent_dim,), name=f"{self.name}_decoder_in")
        decoded = Dense(self.encoding_dim, activation='gelu')(inp)
        decoded = Dense(self.encoding_dim, activation='gelu')(decoded)
        decoded = LayerNormalization()(decoded)
        decoded = Dense(1, activation='linear')(decoded)

        if self.optional:
            optional_decoded = Dense(1, activation='sigmoid')(inp)
            decoded = tf.keras.layers.Concatenate()([decoded, optional_decoded])

        return tf.keras.Model(inp, decoded, name=f"{self.name}_decoder")

    def print_stats(self):
        t = PrettyTable([f"Scalar Field", self.name])
        t.add_row(["Count", self.n])
        t.add_row(["Min", self.min_val])
        t.add_row(["Max", self.max_val])
        t.add_row(["Mean", self.mean_val])
        t.add_row(["Std", self.std_val])
        print(t)


class TextSwapNoise(tf.keras.layers.Layer):
    def __init__(self, vocab_size, swap_prob, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.swap_prob = swap_prob

    def call(self, inputs, training=None):
        # Ensure training is a boolean; if None, default to False.
        training = tf.cast(training if training is not None else False, tf.bool)

        def add_noise():
            # inputs shape: (batch, seq_length)
            shape = tf.shape(inputs)
            # For each token, sample a random float [0,1)
            random_vals = tf.random.uniform(shape, 0, 1, dtype=tf.float32)
            # Create a mask: True where a token should be swapped.
            swap_mask = tf.cast(random_vals < self.swap_prob, inputs.dtype)
            # Sample random tokens for every position.
            random_tokens = tf.random.uniform(
                shape, minval=0, maxval=self.vocab_size, dtype=inputs.dtype
            )
            # For positions marked in swap_mask, replace the token.
            return tf.where(tf.equal(swap_mask, 1), random_tokens, inputs)

        return tf.cond(training, add_noise, lambda: inputs)

    def compute_output_shape(self, input_shape):
        # The noise layer leaves the shape unchanged.
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "swap_prob": self.swap_prob,
        })
        return config
    
    def output_shape(self):
        return self.input_shape


################################################################################
# TextField
################################################################################
class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.05, **kwargs):
        super().__init__(**kwargs)
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='gelu'),
            Dense(embed_dim)
        ])
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        seq_len = tf.shape(x)[1]
        # Create a band mask: each token attends to itself and one token to each side
        # This results in a (seq_len, seq_len) mask where positions more than 1 apart are blocked.
        local_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), 1, 1)
        attn_output = self.att(x, x, attention_mask=local_mask)
        attn_output = self.dropout(attn_output, training=training)
        out1 = self.norm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout(ffn_output, training=training)
        return self.norm2(out1 + ffn_output)

class TransformerDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.05, **kwargs):
        super().__init__(**kwargs)
        self.self_att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='gelu'),
            Dense(embed_dim)
        ])
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        seq_len = tf.shape(x)[1]
        # Create a band mask so that each position attends only to its adjacent tokens.
        local_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), 1, 1)
        attn_output = self.self_att(x, x, attention_mask=local_mask)
        attn_output = self.dropout(attn_output, training=training)
        out1 = self.norm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout(ffn_output, training=training)
        return self.norm2(out1 + ffn_output)
    

class AttentionPooling1D(tf.keras.layers.Layer):
    def __init__(self, feature_dim, num_heads=2, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.supports_masking = True

        self.key_dense = Dense(self.feature_dim * self.num_heads, kernel_initializer="glorot_uniform", name="key_dense")
        self.value_dense = Dense(self.feature_dim * self.num_heads, kernel_initializer="glorot_uniform", name="value_dense")
        self.output_dense = Dense(self.feature_dim, kernel_initializer="glorot_uniform", name="output_dense")
        self.attention_dropout = Dropout(self.dropout_rate)

        self.global_query = self.add_weight(
            shape=(self.num_heads, self.feature_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="global_query",
        )

    def call(self, inputs, mask=None, training=None):
        batch_size = tf.shape(inputs)[0]
        keys = self.key_dense(inputs)
        values = self.value_dense(inputs)

        keys = tf.reshape(keys, (batch_size, -1, self.num_heads, self.feature_dim))
        values = tf.reshape(values, (batch_size, -1, self.num_heads, self.feature_dim))
        keys = tf.transpose(keys, [0, 2, 1, 3])
        values = tf.transpose(values, [0, 2, 1, 3])

        query = tf.expand_dims(self.global_query, axis=0)
        query = tf.tile(query, [batch_size, 1, 1])
        query = tf.expand_dims(query, axis=2)

        scores = tf.matmul(query, keys, transpose_b=True)
        scores /= tf.math.sqrt(tf.cast(self.feature_dim, scores.dtype))

        if mask is not None:
            mask = tf.cast(tf.reshape(mask, [batch_size, 1, 1, -1]), scores.dtype)
            scores += (1.0 - mask) * -1e9

        weights = tf.nn.softmax(scores, axis=-1)
        weights = self.attention_dropout(weights, training=training)
        pooled = tf.matmul(weights, values)
        pooled = tf.squeeze(pooled, axis=2)
        pooled = tf.reshape(pooled, (batch_size, self.num_heads * self.feature_dim))
        output = self.output_dense(pooled)
        return output

class GlobalSummaryPooling1D(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs):
        # inputs shape: (batch, sequence_length, channels)
        mean = tf.reduce_mean(inputs, axis=1)          # (batch, channels)
        max_val = tf.reduce_max(inputs, axis=1)         # (batch, channels)
        min_val = tf.reduce_min(inputs, axis=1)         # (batch, channels)
        # Standard deviation calculation
        std = tf.sqrt(tf.reduce_mean(tf.square(inputs - tf.expand_dims(mean, axis=1)), axis=1) + self.epsilon)
        # Range: difference between max and min
        range_val = max_val - min_val
        # Skewness: E[(x - mean)^3] / (std^3 + epsilon)
        skew_numer = tf.reduce_mean(tf.pow(inputs - tf.expand_dims(mean, axis=1), 3), axis=1)
        skew = skew_numer / (tf.pow(std, 3) + self.epsilon)
        
        # Stack metrics along a new axis, resulting in shape (batch, 6, channels)
        summary = tf.stack([max_val, mean, std, min_val, range_val, skew], axis=1)
        return summary

    def compute_output_shape(self, input_shape):
        # Input shape is (batch, sequence_length, channels)
        return (input_shape[0], 6, input_shape[2])
    

class GlobalMaskedPooling1D(tf.keras.layers.Layer):
    def __init__(self, input_size, epsilon=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        # This conv will reduce the stats dimension (of length 6) to 1.
        self.mask_conv_big = tf.keras.layers.Conv1D(filters=64, kernel_size=1, padding="same", activation="linear")
        self.mask_conv = tf.keras.layers.Conv1D(filters=1, kernel_size=1, padding="same", activation="gelu")
        self.mask_dense = tf.keras.layers.Dense(input_size*2, activation="linear")
        self.mask_dense_final = tf.keras.layers.Dense(input_size, activation="sigmoid")
        self.flatten = Flatten()
        self.softmax = Softmax()

    def call(self, inputs):
        # inputs shape: (batch, sequence_length, channels)
        max_val = tf.reduce_max(inputs, axis=1, keepdims=True)  # (batch, 1, channels)
        mean_val = tf.reduce_mean(inputs, axis=1, keepdims=True)  # (batch, 1, channels)
        std_val = tf.sqrt(tf.reduce_mean(tf.square(inputs - mean_val), axis=1, keepdims=True) + self.epsilon)  # (batch, 1, channels)

        # Skewness: E[(x - mean)^3] / (std^3 + epsilon)
        skew = tf.reduce_mean(tf.pow(inputs - mean_val, 3), axis=1, keepdims=True) / (tf.pow(std_val, 3) + self.epsilon)  # (batch, 1, channels)

        # Entropy: treat the sequence values as logits over positions per channel.
        # Apply softmax along the sequence dimension to get probabilities.
        p = tf.nn.softmax(inputs, axis=1)
        entropy = -tf.reduce_sum(p * tf.math.log(p + self.epsilon), axis=1, keepdims=True)  # (batch, 1, channels)

        # Kurtosis: E[(x - mean)^4] / (std^4 + epsilon) - 3.
        kurtosis = tf.reduce_mean(tf.pow(inputs - mean_val, 4), axis=1, keepdims=True) / (tf.pow(std_val, 4) + self.epsilon) - 3  # (batch, 1, channels)

        # Stack the metrics into a tensor of shape (batch, 6, channels)
        # We are including: max, mean, std, skew, entropy, kurtosis.
        stats = tf.concat([max_val, mean_val, std_val, skew, entropy, kurtosis], axis=1)
        # Rotate so that the channels become the steps dimension: (batch, channels, 6)
        stats = tf.transpose(stats, perm=[0, 2, 1])
        
        # Apply a Conv1D along the stats dimension to produce a mask per channel.
        # The conv will reduce the 6 stats to 1.
        mask = self.mask_conv_big(stats)
        mask = self.mask_conv(mask)
        mask = self.flatten(mask)
        mask = self.mask_dense(mask)
        mask = self.mask_dense_final(mask)
        mask = tf.expand_dims(mask, axis=1)
        
        # Multiply the mask by the maxpooled vector.
        output = mask * max_val
        return output

    def compute_output_shape(self, input_shape):
        # Output shape: (batch, 1, channels)
        return (input_shape[0], 1, input_shape[2])



@tf.keras.utils.register_keras_serializable(package="Custom", name="MaskedSparseCategoricalCrossentropy")
class MaskedSparseCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, char_to_index, **kwargs):
        super().__init__(**kwargs)
        self.char_to_index = char_to_index

    def call(self, y_true, y_pred):
        pad_token_index = self.char_to_index[SPECIAL_PAD]
        mask = tf.cast(tf.not_equal(y_true, pad_token_index), dtype=tf.float32)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        masked_loss = loss * mask
        return tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)

    def get_config(self):
        config = super().get_config()
        config.update({
            "char_to_index": self.char_to_index,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TextField(BaseField):
    def __init__(
        self,
        name: str,
        max_length=None,
        downsample_steps=2,
        base_size=64,
        num_blocks_per_step=[2, 2],
        optional: bool = False
    ):
        super().__init__(name, optional)
        self.user_max_length = max_length
        self.dynamic_max_len = 0
        self.alphabet = set()
        self.char_to_index = {}
        self.index_to_char = {}
        self.max_length = None
        self.downsample_steps = downsample_steps
        self.base_size = base_size
        self.num_blocks_per_step = num_blocks_per_step

        if optional:
            raise ValueError("TextField does not support optional fields.")

    def _get_input_shape(self):
        return (self.max_length,)

    def _get_output_shape(self):
        return (self.max_length,)

    def _get_loss(self):
        return MaskedSparseCategoricalCrossentropy(self.char_to_index)
    
    def _get_weight(self):
        return 2.0

    def _accumulate_stats(self, raw_value):
        if raw_value:
            txt = str(raw_value)
            self.dynamic_max_len = max(self.dynamic_max_len, len(txt))
            for char in txt:
                self.alphabet.add(char)

    def _finalize_stats(self):
        sorted_alphabet = sorted(list(self.alphabet))
        full_alphabet = [SPECIAL_PAD, SPECIAL_START, SPECIAL_END, SPECIAL_SEP] + sorted_alphabet
        self.char_to_index = {char: idx for idx, char in enumerate(full_alphabet)}
        self.index_to_char = {idx: char for char, idx in self.char_to_index.items()}

        raw_len = self.dynamic_max_len + 2  # +2 for start/end tokens
        if self.user_max_length is not None:
            self.max_length = self.user_max_length
            if raw_len > self.user_max_length:
                print(f"Warning: user_max_length ({self.user_max_length}) < needed ({raw_len}); truncation may occur.")
        else:
            self.max_length = raw_len

        multiple = 2 ** self.downsample_steps
        if multiple > 1:
            rounded_len = ((self.max_length + multiple - 1) // multiple) * multiple
            if rounded_len != self.max_length:
                print(f"Adjusting max_length from {self.max_length} to {rounded_len} for divisibility by {multiple}.")
                self.max_length = rounded_len

    def _transform(self, raw_value):
        if not raw_value:
            txt = ""
        else:
            txt = str(raw_value)
        tokens = [self.char_to_index[SPECIAL_START]] + [
            self.char_to_index.get(c, self.char_to_index[SPECIAL_PAD]) for c in txt
        ] + [self.char_to_index[SPECIAL_END]]

        tokens = tokens[:self.max_length]
        pad_needed = self.max_length - len(tokens)
        if pad_needed > 0:
            tokens += [self.char_to_index[SPECIAL_PAD]] * pad_needed

        return tf.constant(tokens, dtype=tf.int32)

    def to_string(self, predicted_array: np.ndarray) -> str:
        if len(predicted_array.shape) == 3:
            predicted_array = predicted_array[0]

        probs = predicted_array
        token_indices = probs.argmax(axis=-1)
        tokens = [self.index_to_char.get(idx, '') for idx in token_indices]

        out_chars = []
        for t in tokens:
            if t == SPECIAL_END:
                break
            if t not in (SPECIAL_START, SPECIAL_PAD, SPECIAL_END):
                out_chars.append(t)
        return "".join(out_chars)

    def build_encoder(self, latent_dim: int) -> tf.keras.Model:
        inp = tf.keras.Input(shape=(self.max_length,), name=f"{self.name}_input")
        embedding = Embedding(
            input_dim=len(self.char_to_index),
            output_dim=self.base_size,
            name=f"{self.name}_embedding"
        )(inp)

        x = Lambda(add_positional_encoding,
                output_shape=lambda s: (s[0], s[1], s[2]),
                name="add_positional_encoding")(embedding)

        current_length = self.max_length
        current_filters = self.base_size

        for step in range(self.downsample_steps):
            num_blocks = self.num_blocks_per_step[step] if step < len(self.num_blocks_per_step) else 1
            for block in range(num_blocks):
                x = self._residual_block(x, current_filters, block_num=block, step_num=step)
            current_length //= 2
            current_filters *= 2
            x = Conv1D(current_filters, 3, strides=2, activation='gelu', padding='same')(x)

        x = Conv1D(current_filters, 3, activation='linear')(x)
        x = Flatten()(x)
        return tf.keras.Model(inp, x, name=f"{self.name}_encoder")

    def build_decoder(self, latent_dim: int) -> tf.keras.Model:
        inp = tf.keras.Input(shape=(latent_dim,), name=f"{self.name}_decoder_in")
        current_length = self.max_length // (2 ** self.downsample_steps)
        current_filters = self.base_size * (2 ** self.downsample_steps)

        x = Dense(current_length * self.base_size, activation='gelu')(inp)
        x = Reshape((current_length, self.base_size))(x)
        x = Lambda(add_positional_encoding, 
                output_shape=lambda s: (s[0], s[1], s[2]),
                name="add_positional_encoding")(x)

        for step in range(self.downsample_steps):
            x = Conv1DTranspose(current_filters * 2, 3, strides=2, activation='gelu', padding='same')(x)
            num_blocks = self.num_blocks_per_step[step] if step < len(self.num_blocks_per_step) else 1
            for block in range(num_blocks):
                x = self._residual_block(x, current_filters, block_num=block, step_num=step)
            current_filters //= 2
            current_length *= 2

        out = Conv1D(len(self.char_to_index), 1, activation='linear')(x)
        out = Softmax()(out)
        return tf.keras.Model(inp, out, name=f"{self.name}_decoder")

    def _residual_block(self, x, filters, reduction_ratio=2, kernel_size=3, block_num=None, step_num=None):
        if block_num is not None and step_num is not None:
            print(f"Applying Residual Block {block_num + 1} at Step {step_num + 1} with {filters} filters.")

        shortcut = x
        x = Conv1D(filters // reduction_ratio, 1, activation='gelu', padding='same')(x)
        x = Conv1D(filters // reduction_ratio, kernel_size, activation='gelu', padding='same')(x)
        x = Conv1D(filters, 1, activation='linear', padding='same')(x)
        if shortcut.shape[-1] != x.shape[-1]:
            shortcut = Conv1D(x.shape[-1], 1, padding='same', name=f'adjustment_conv_S{step_num}_B{block_num}')(shortcut)
        return Add()([x, shortcut])

    def print_stats(self):
        t = PrettyTable([f"Text Field", self.name])
        t.add_row(["Unique chars", len(self.alphabet)])
        t.add_row(["Max raw length", self.dynamic_max_len])
        t.add_row(["Final max_length", self.max_length])
        print(t)

class MultiCategoryToggleLayer(Layer):
    """
    Flips each one-hot bit with probability `p` only when `training=True`.
    """
    def __init__(self, p, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def call(self, inputs, training=None):
        # If `training` is None, default to False. (This is typical for
        # a functional model call without an explicit 'training' arg.)
        if training is None:
            training = False

        def no_noise():
            return inputs

        def add_noise():
            random_vals = tf.random.uniform(tf.shape(inputs), 0, 1, dtype=inputs.dtype)
            toggles = tf.cast(random_vals < self.p, inputs.dtype)
            return tf.where(toggles > 0, 1.0 - inputs, inputs)

        # tf.cond expects boolean scalar for condition
        return tf.cond(tf.cast(training, tf.bool), add_noise, no_noise)

    def compute_output_shape(self, input_shape):
        # Same shape as the input
        return input_shape
    
    def compute_output_signature(self, input_signature):
        # Same dtype/shape as input
        return input_signature

################################################################################
# MultiCategoryField
################################################################################

class MultiCategoryField(BaseField):
    def __init__(self, name: str, optional: bool = False):
        super().__init__(name, optional)
        self.category_set = set()
        self.category_list = []

    def _get_input_shape(self):
        return (len(self.category_list),)

    def _get_output_shape(self):
        return (len(self.category_list),)

    def _get_loss(self):
        return tf.keras.losses.BinaryCrossentropy()

    def _accumulate_stats(self, raw_value):
        if raw_value:
            cats = raw_value if isinstance(raw_value, list) else [raw_value]
            for c in cats:
                self.category_set.add(str(c))

    def _finalize_stats(self):
        self.category_list = sorted(self.category_set)

    def _transform(self, raw_value):
        cats = raw_value if isinstance(raw_value, list) else ([raw_value] if raw_value else [])
        vec = np.zeros(len(self.category_list), dtype=np.float32)
        for c in cats:
            c_str = str(c)
            if c_str in self.category_list:
                idx = self.category_list.index(c_str)
                vec[idx] = 1.0
        return tf.constant(vec, dtype=tf.float32)
    
    def noise_function(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        return MultiCategoryToggleLayer(p=0.05)(x)

    def _toggle_categories(self, cat_values: tf.Tensor, p: float) -> tf.Tensor:
        """
        Helper that flips bits in cat_values with probability p.
        cat_values shape: (batch, num_categories)
        """
        random_vals = tf.random.uniform(tf.shape(cat_values), 0, 1, dtype=cat_values.dtype)
        toggles = tf.cast(random_vals < p, cat_values.dtype)  # 1 where we flip, else 0
        # Flipping means: new_val = 1 - old_val
        # We'll do tf.where(toggles > 0, 1 - x, x)
        return tf.where(toggles > 0, 1.0 - cat_values, cat_values)

    def to_string(self, predicted_array: np.ndarray) -> str:
        if self.optional:
            vec_part = predicted_array[:-1]
            is_null_part = predicted_array[-1]
            if is_null_part > 0.4:
                return "None"
            probs = vec_part
        else:
            probs = predicted_array

        threshold = 0.4

        selected = []
        for i, p in enumerate(probs):
            if p >= threshold:
                selected.append(self.category_list[i])
        return ', '.join(selected) if selected else "(none)"

    def build_encoder(self, latent_dim: int) -> tf.keras.Model:
        inp = tf.keras.Input(shape=self.input_shape, name=f"{self.name}_input", dtype=self.input_dtype)
        x = self.noise_function(inp)
        x = Dense(32, activation='gelu')(inp)
        x = LayerNormalization()(x)
        x = Dense(32, activation='gelu')(x)
        out = LayerNormalization()(x)
        return tf.keras.Model(inp, out, name=f"{self.name}_encoder")

    def build_decoder(self, latent_dim: int) -> tf.keras.Model:
        inp = tf.keras.Input(shape=(latent_dim,), name=f"{self.name}_decoder_in")
        
        x = Dense(32, activation='gelu')(inp)
        x = LayerNormalization()(x)
        x = Dense(32, activation='gelu')(x)
        x = LayerNormalization()(x)
        # no denoising residual
        out = Dense(self.output_shape[0], activation='sigmoid', name=f"{self.name}_decoder")(x)
        return tf.keras.Model(inp, out, name=f"{self.name}_decoder")

    def print_stats(self):
        t = PrettyTable([f"MultiCategory Field", self.name])
        t.add_row(["Unique categories", len(self.category_list)])
        for i, c in enumerate(self.category_list):
            t.add_row([i, c])
        print(t)
