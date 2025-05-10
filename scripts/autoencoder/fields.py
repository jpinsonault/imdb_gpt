import logging
import math
from typing import List, Optional
from prettytable import PrettyTable
import numpy as np
import tensorflow as tf
from enum import Enum
from tensorflow.keras.layers import (
    Dense,
    Embedding,
    LayerNormalization,
    Reshape,
    Softmax,
    Dropout,
    CategoryEncoding,
    Lambda,
    Flatten,
    Conv1D,
    Activation,
    Conv1DTranspose, # Added Concatenate
)
import abc
from .character_tokenizer import CharacterTokenizer
################################################################################
# Enums and constants
################################################################################

# Configure logging (optional, but good practice for the custom tokenizer)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        self._stats_finalized = False

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
        """Transforms a non-None raw value into its base tensor representation."""
        pass

    @abc.abstractmethod
    def build_encoder(self, latent_dim: int) -> tf.keras.Model:
        pass

    @abc.abstractmethod
    def build_decoder(self, latent_dim: int) -> tf.keras.Model:
        """Builds the decoder. If optional, MUST return a Model with TWO outputs: [main_output, flag_output]."""
        pass

    @abc.abstractmethod
    def to_string(self, predicted_main: np.ndarray, predicted_flag: Optional[np.ndarray] = None) -> str:
         """Converts prediction tensor(s) back to a string representation."""
         # Note: Signature changed to accept separate flag if field is optional.
         # Implementation needs adjustment in subclasses.
         pass

    @abc.abstractmethod
    def print_stats(self):
        pass

    @abc.abstractmethod
    def get_base_padding_value(self):
        """Returns the padding value for the main part of the field."""
        pass

    @abc.abstractmethod
    def get_flag_padding_value(self):
        """Returns the padding value for the flag part (typically 1.0 for 'is_null')."""
        pass

    @property
    def input_shape(self):
        # Input shape only considers the base features, not the flag.
        return self._get_input_shape()

    @property
    def input_dtype(self):
        # Default dtype, override in subclasses if needed (e.g., TextField).
        return tf.float32

    @property
    def output_shape(self):
         # Base shape of the main output head.
         # The flag output head shape is always (1,).
         return self._get_output_shape()

    @property
    def output_dtype(self):
         # Default dtype for main output, override if needed.
         # Flag output is always float32.
         return tf.float32

    @property
    def weight(self):
        return self._get_weight()

    def accumulate_stats(self, raw_value):
        self._accumulate_stats(raw_value)

    def finalize_stats(self):
        self._finalize_stats()
        self._stats_finalized = True
    
    def stats_finalized(self) -> bool:
        return self._stats_finalized
    
    def transform(self, raw_value):
        """Transforms a raw value for model INPUT. Handles None for optional fields."""
        if raw_value is None:
            if not self.optional:
                raise ValueError(f"Field '{self.name}' is not optional, but received None.")
            # Use the base padding value as input when raw value is None.
            return self.get_base_padding_value()
        else:
            # Transform the actual value.
            return self._transform(raw_value)

    def transform_target(self, raw_value):
        """Transforms a raw value into TARGET tensor(s) for model training/evaluation."""
        if raw_value is None:
            if not self.optional:
                # This check is still valid for handling None input for non-optional fields
                raise ValueError(f"Field '{self.name}' is not optional, but received None.")
            # Optional field and None value: Target is base padding value and flag = 1.0
            main_target = self.get_base_padding_value()
            flag_target = self.get_flag_padding_value() # Should be 1.0
            return main_target, flag_target # Return tuple ONLY for optional fields when value is None
        else:
            # Value is not None: Target is transformed value. Flag depends on optionality.
            main_target = self._transform(raw_value)
            if self.optional:
                # If optional, return the flag indicating 'not null' (0.0)
                flag_target = 1.0 - self.get_flag_padding_value() # Gets 0.0
                return main_target, flag_target # Return tuple
            else:
                # If not optional, just return the main target value
                return main_target # Return single tensor

    @property
    def loss(self):
        """Returns the loss function for the main output head."""
        # No longer uses _optional_loss_wrapper.
        # Returns the base loss directly.
        # A separate loss (BCE) will be applied to the flag output head.
        return self._get_loss()

    def get_flag_loss(self):
        """Returns the loss function for the optional flag output head."""
        if not self.optional:
            return None
        return tf.keras.losses.BinaryCrossentropy()


    def _get_weight(self):
        # Default weight, can be overridden.
        # Consider if flag loss needs separate weighting.
        return 1.0

################################################################################
# BooleanField
################################################################################

class BooleanField(BaseField):
    def __init__(self, name: str, use_bce_loss: bool = True, optional: bool = False):
        super().__init__(name, optional)
        self.use_bce_loss = use_bce_loss
        self.count_total = 0
        self.count_ones = 0

    def _get_input_shape(self):
        return (1,)

    def _get_output_shape(self):
        return (1,)

    def _get_loss(self):
        if self.use_bce_loss:
            return tf.keras.losses.BinaryCrossentropy()
        else:
            return tf.keras.losses.MeanSquaredError()

    def get_base_padding_value(self):
        return tf.constant([0.0], dtype=tf.float32)

    def get_flag_padding_value(self):
         return tf.constant([1.0], dtype=tf.float32)

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
        pass

    def _transform(self, raw_value):
        try:
            val = float(raw_value)
        except (ValueError, TypeError):
             val = 0.0 # Default if transformation fails
        val = 1.0 if val == 1.0 else 0.0
        return tf.constant([val], dtype=tf.float32)

    def to_string(self, predicted_main: np.ndarray, predicted_flag: Optional[np.ndarray] = None) -> str:
        if self.optional:
             if predicted_flag is None:
                  raise ValueError("predicted_flag is required for optional BooleanField.to_string")
             is_null_prob = float(predicted_flag[0])
             if is_null_prob > 0.5:
                 return "None"
             prob_true = float(predicted_main[0])
        else:
            prob_true = float(predicted_main[0])
        return "True" if prob_true >= 0.5 else "False"

    def build_encoder(self, latent_dim: int) -> tf.keras.Model:
        # Encoder input shape uses self.input_shape which is base shape only
        inp = tf.keras.Input(shape=self.input_shape, name=f"{self.name}_input", dtype=self.input_dtype)
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
        main_out = Dense(self._get_output_shape()[0], activation='sigmoid', name=f"{self.name}_main_out")(x)

        if self.optional:
            flag_out = Dense(1, activation='sigmoid', name=f"{self.name}_flag_out")(x)
            return tf.keras.Model(inp, [main_out, flag_out], name=f"{self.name}_decoder")
        else:
            return tf.keras.Model(inp, main_out, name=f"{self.name}_decoder")


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
        self.n = 0
        self.sum_ = 0.0
        self.sum_sq = 0.0
        self.min_val = float("inf")
        self.max_val = float("-inf")
        self.mean_val = 0.0
        self.std_val = 1.0
        self.encoding_dim = 8

    def _get_input_shape(self):
        return (1,)

    def _get_output_shape(self):
        return (1,)

    def _get_loss(self):
        return tf.keras.losses.MeanSquaredError()

    def get_base_padding_value(self):
         # Padding value depends on scaling, return the scaled representation of 0.0
         zero_val = 0.0
         if self.scaling == Scaling.NONE:
             transformed = self.scaling.none_transform(zero_val)
         elif self.scaling == Scaling.NORMALIZE:
             # Handle finalize_stats potentially not being called yet
             min_v = self.min_val if self.min_val != float("inf") else 0.0
             max_v = self.max_val if self.max_val != float("-inf") else 0.0
             transformed = self.scaling.normalize_transform(zero_val, min_val=min_v, max_val=max_v)
         elif self.scaling == Scaling.STANDARDIZE:
             mean_v = self.mean_val if self.n > 0 else 0.0
             std_v = self.std_val if self.n > 0 else 1.0
             transformed = self.scaling.standardize_transform(zero_val, mean_val=mean_v, std_val=std_v)
         elif self.scaling == Scaling.LOG:
             transformed = self.scaling.log_transform(zero_val)
         return tf.constant([transformed], dtype=tf.float32)

    def get_flag_padding_value(self):
         return tf.constant([1.0], dtype=tf.float32)

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
            # Ensure variance is non-negative for sqrt
            self.std_val = np.sqrt(max(0, var)) if max(0, var) > 1e-12 else 1.0
            if self.std_val == 0: self.std_val = 1.0 # Avoid division by zero
        else:
            self.min_val = 0.0
            self.max_val = 0.0
            self.mean_val = 0.0
            self.std_val = 1.0


    def noise_function(self, x):
        gaussian_noise = tf.keras.layers.GaussianNoise(stddev=0.05)(x)
        return x + gaussian_noise

    def _transform(self, raw_value):
        try:
            x = float(raw_value)
        except (ValueError, TypeError):
            x = 0.0 # Default value
        if self.clip_max is not None:
            x = min(x, self.clip_max)

        # Apply scaling function as needed:
        if self.scaling == Scaling.NONE:
            transformed = self.scaling.none_transform(x)
        elif self.scaling == Scaling.NORMALIZE:
            transformed = self.scaling.normalize_transform(x, min_val=self.min_val, max_val=self.max_val)
        elif self.scaling == Scaling.STANDARDIZE:
            transformed = self.scaling.standardize_transform(x, mean_val=self.mean_val, std_val=self.std_val)
        elif self.scaling == Scaling.LOG:
            transformed = self.scaling.log_transform(x)
        return tf.constant([transformed], dtype=tf.float32)

    def to_string(self, predicted_main: np.ndarray, predicted_flag: Optional[np.ndarray] = None) -> str:
        if self.optional:
             if predicted_flag is None:
                  raise ValueError("predicted_flag is required for optional ScalarField.to_string")
             is_null_prob = float(predicted_flag[0])
             if is_null_prob > 0.5:
                 return "None"
             float_val = float(predicted_main[0])
        else:
             float_val = float(predicted_main[0])

        # Untransform
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
        inp = tf.keras.Input(shape=self.input_shape, name=f"{self.name}_input", dtype=self.input_dtype)
        # inp_noisy = self.noise_function(inp) # Apply noise after input layer if desired
        return tf.keras.Model(inp, inp, name=f"{self.name}_encoder")

    def build_decoder(self, latent_dim: int) -> tf.keras.Model:
        inp = tf.keras.Input(shape=(latent_dim,), name=f"{self.name}_decoder_in")
        x = Dense(8, activation='gelu')(inp)
        x = LayerNormalization()(x)
        x = Dense(8, activation='gelu')(x)
        x = LayerNormalization()(x)
        main_out = Dense(self._get_output_shape()[0], activation='linear', name=f"{self.name}_main_out")(x)

        if self.optional:
            flag_out = Dense(1, activation='sigmoid', name=f"{self.name}_flag_out")(x)
            return tf.keras.Model(inp, [main_out, flag_out], name=f"{self.name}_decoder")
        else:
            return tf.keras.Model(inp, main_out, name=f"{self.name}_decoder")

    def print_stats(self):
        t = PrettyTable([f"Scalar Field", self.name])
        t.add_row(["Count", self.n])
        t.add_row(["Min", f"{self.min_val:.4f}" if self.min_val != float("inf") else "N/A"])
        t.add_row(["Max", f"{self.max_val:.4f}" if self.max_val != float("-inf") else "N/A"])
        t.add_row(["Mean", f"{self.mean_val:.4f}" if self.n > 0 else "N/A"])
        t.add_row(["Std", f"{self.std_val:.4f}" if self.n > 0 else "N/A"])
        print(t)


class TextSwapNoise(tf.keras.layers.Layer):
    def __init__(self, vocab_size, swap_prob, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.swap_prob = swap_prob

    def call(self, inputs, training=None):
        training = tf.cast(training if training is not None else False, tf.bool)

        def add_noise():
            shape = tf.shape(inputs)
            random_vals = tf.random.uniform(shape, 0, 1, dtype=tf.float32)
            swap_mask = tf.cast(random_vals < self.swap_prob, inputs.dtype)
            random_tokens = tf.random.uniform(
                shape, minval=0, maxval=self.vocab_size, dtype=inputs.dtype
            )
            return tf.where(tf.equal(swap_mask, 1), random_tokens, inputs)

        return tf.cond(training, add_noise, lambda: inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "swap_prob": self.swap_prob,
        })
        return config

    def output_shape(self):
         # Keras 3 requires output_shape to be defined if compute_output_shape is
         # However, TextSwapNoise input shape determines output shape.
         # This might need refinement based on exact Keras version behavior.
         # For now, let's assume it can be inferred or return None/raise.
         # raise NotImplementedError("output_shape cannot be determined statically for TextSwapNoise")
         # Or, if you know the input shape beforehand (less flexible):
         # return self._build_input_shape
         # Safest is likely to rely on compute_output_shape
         return None # Let Keras infer using compute_output_shape


################################################################################
# MaskedSparseCategoricalCrossentropy Loss
################################################################################
@tf.keras.utils.register_keras_serializable(package="Custom", name="MaskedSparseCategoricalCrossentropy")
class MaskedSparseCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, ignore_class: int, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name="masked_sparse_categorical_crossentropy"):
        super().__init__(reduction=reduction, name=name)
        self.ignore_class = int(ignore_class)

    def call(self, y_true, y_pred, sample_weight=None):
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=False, axis=-1
        )
        mask = tf.cast(tf.not_equal(y_true, self.ignore_class), loss.dtype)
        masked_loss = loss * mask
        per_timestep_loss = tf.reduce_mean(masked_loss, axis=-1)
        per_sample_loss = tf.reduce_mean(per_timestep_loss, axis=-1)

        if sample_weight is not None:
            per_sample_loss *= sample_weight

        return per_sample_loss

    def get_config(self):
        config = super().get_config()
        config.update({"ignore_class": self.ignore_class})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


################################################################################
# TextField
################################################################################


class TextField(BaseField):
    def __init__(
        self,
        name: str,
        max_length: Optional[int] = None,
        downsample_steps: int = 2,
        base_size: int = 48,
        num_blocks_per_step: List[int] = [2, 2],
        optional: bool = False
    ):
        super().__init__(name, optional=optional)
        self.user_max_length = max_length
        self.downsample_steps = downsample_steps
        self.base_size = base_size
        self.num_blocks_per_step = num_blocks_per_step
        self.texts: List[str] = []
        self.dynamic_max_len: int = 0
        self.tokenizer: Optional[CharacterTokenizer] = None
        self.max_length: Optional[int] = None
        self.pad_token_id: Optional[int] = None
        self.null_token_id: Optional[int] = None

        self.avg_raw_length: Optional[float] = None
        self.avg_token_count: Optional[float] = None
        self.avg_chars_saved: Optional[float] = None
        self.compression_ratio: Optional[float] = None

    @property
    def input_dtype(self):
        return tf.int32

    @property
    def output_dtype(self):
        return tf.int32

    def _get_input_shape(self):
        if self.max_length is None:
            raise ValueError("TextField stats not finalized. Call finalize_stats() first.")
        return (self.max_length,)

    def _get_output_shape(self):
        if self.max_length is None:
            raise ValueError("TextField stats not finalized. Call finalize_stats() first.")
        return (self.max_length,)

    def _get_loss(self):
        if self.pad_token_id is None:
            raise ValueError("TextField stats not finalized or PAD token ID not set.")
        return MaskedSparseCategoricalCrossentropy(ignore_class=self.pad_token_id)

    def get_flag_loss(self):
        return None

    def get_base_padding_value(self):
        if self.pad_token_id is None or self.max_length is None:
            raise RuntimeError("TextField stats not finalized. Call finalize_stats() first.")
        return tf.constant([self.pad_token_id] * self.max_length, dtype=tf.int32)

    def get_flag_padding_value(self):
        return tf.constant([1.0], dtype=tf.float32)

    def _accumulate_stats(self, raw_value):
        if raw_value is not None:
            txt = str(raw_value)
            if txt:
                self.texts.append(txt)

    def _finalize_stats(self):
        # Build the list of special tokens.
        special_tokens = [SPECIAL_PAD, SPECIAL_START, SPECIAL_END]

        if not self.texts and self.optional:
            logging.warning(f"No text data for optional field '{self.name}'. Training tokenizer only on special tokens.")
            self.tokenizer = CharacterTokenizer(special_tokens=special_tokens)
            self.tokenizer.train([])
        elif not self.texts and not self.optional:
            logging.error(f"No text data accumulated for non-optional field '{self.name}'.")
            self.tokenizer = CharacterTokenizer(special_tokens=special_tokens)
            self.tokenizer.train([])
        else:
            logging.info(f"Finalizing stats for TextField '{self.name}'. Training character tokenizer...")
            self.tokenizer = CharacterTokenizer(special_tokens=special_tokens)
            self.tokenizer.train(self.texts)

        self.pad_token_id = self.tokenizer.token_to_id(SPECIAL_PAD)
        if self.pad_token_id is None:
            raise ValueError(f"Could not find PAD token '{SPECIAL_PAD}' in the trained tokenizer vocab.")

        max_tokens = 0
        total_raw_chars = 0
        total_tokens = 0
        num_texts = len(self.texts)
        if num_texts > 0:
            for txt in self.texts:
                token_ids = self.tokenizer.encode(txt)
                token_count = len(token_ids)
                total_tokens += token_count
                total_raw_chars += len(txt)
                max_tokens = max(max_tokens, token_count)
            self.avg_raw_length = total_raw_chars / num_texts
            self.avg_token_count = total_tokens / num_texts
            self.avg_chars_saved = self.avg_raw_length - self.avg_token_count
            self.compression_ratio = (self.avg_raw_length / self.avg_token_count) if self.avg_token_count > 0 else None
        else:
            self.avg_raw_length = 0.0
            self.avg_token_count = 0.0
            self.avg_chars_saved = 0.0
            self.compression_ratio = None

        self.dynamic_max_len = max_tokens
        effective_max_len = max_tokens

        if self.user_max_length is not None:
            self.max_length = self.user_max_length
            if effective_max_len > self.user_max_length:
                logging.warning(
                    f"User-defined max_length ({self.user_max_length}) is less than the required length ({effective_max_len}) for field '{self.name}'. Sequences will be truncated."
                )
        else:
            self.max_length = effective_max_len

        self.max_length = max(1, self.max_length)

        multiple = 2 ** self.downsample_steps
        if multiple > 1:
            original_max_length = self.max_length
            adjusted_len = max(multiple, self.max_length)
            rounded_len = ((adjusted_len + multiple - 1) // multiple) * multiple
            if rounded_len != original_max_length:
                logging.info(
                    f"Adjusting max_length for field '{self.name}' from {original_max_length} to {rounded_len} for divisibility by {multiple} (downsample_steps={self.downsample_steps})."
                )
                self.max_length = rounded_len

        self.print_stats()

    def _transform(self, raw_value):
        txt = str(raw_value)
        token_ids = self.tokenizer.encode(txt)
        start_token_id = self.tokenizer.token_to_id(SPECIAL_START)
        end_token_id = self.tokenizer.token_to_id(SPECIAL_END)
        token_ids = [start_token_id] + token_ids + [end_token_id]
        current_len = len(token_ids)
        if current_len < self.max_length:
            pad_length = self.max_length - current_len
            token_ids += [self.pad_token_id] * pad_length
        else:
            token_ids = token_ids[:self.max_length]
            if token_ids[-1] != end_token_id:
                pass
        return tf.constant(token_ids, dtype=tf.int32)

    def transform_target(self, raw_value):
        if raw_value is None:
            if not self.optional:
                raise ValueError(f"Field '{self.name}' is not optional, but received None.")
            start_token_id = self.tokenizer.token_to_id(SPECIAL_START)
            end_token_id = self.tokenizer.token_to_id(SPECIAL_END)
            null_token_id = self.null_token_id
            token_ids = [start_token_id, null_token_id, end_token_id]
            current_len = len(token_ids)
            if current_len < self.max_length:
                pad_length = self.max_length - current_len
                token_ids += [self.pad_token_id] * pad_length
            else:
                token_ids = token_ids[:self.max_length]
            main_target = tf.constant(token_ids, dtype=tf.int32)
            return main_target
        else:
            main_target = self._transform(raw_value)
            return main_target

    def transform(self, raw_value):
        if raw_value is None:
            if not self.optional:
                raise ValueError(f"Field '{self.name}' is not optional, but received None.")
            return self.get_base_padding_value()
        else:
            return self._transform(raw_value)

    def to_string(self, predicted_main: np.ndarray, predicted_flag: Optional[np.ndarray] = None) -> str:
        token_indices = predicted_main.flatten().astype(int).tolist()
        tokens = [self.tokenizer.id_to_token(idx) for idx in token_indices]
        out_tokens = []
        for token in tokens:
            if token == SPECIAL_END:
                break
            if token in (SPECIAL_START, SPECIAL_PAD):
                continue
            out_tokens.append(token)
        return "".join(out_tokens)

    def build_encoder(self, latent_dim: int) -> tf.keras.Model:
        inp = tf.keras.Input(shape=(self.max_length,), dtype=tf.int32, name=f"{self.name}_input")
        x = Embedding(
                input_dim=self.tokenizer.get_vocab_size(),
                output_dim=self.base_size,
                mask_zero=False,
                name=f"{self.name}_embedding"
            )(inp)
        x = Conv1D(
                self.base_size * 2,
                kernel_size=5,
                strides=2,
                activation='gelu',
                padding='same',
                name=f"{self.name}_conv1"
            )(x)
        x = Flatten()(x)
        token_latent = Dense(latent_dim, activation=None, name=f"{self.name}_to_latent")(x)
        return tf.keras.Model(inp, token_latent, name=f"{self.name}_encoder")

    def build_decoder(self, latent_dim: int) -> tf.keras.Model:
        if self.max_length is None or self.tokenizer is None or self.base_size is None:
            raise RuntimeError(f"Decoder for TextField '{self.name}' cannot be built before stats are finalized.")
        latent_in = tf.keras.Input(shape=(latent_dim,), name=f"{self.name}_decoder_latent_in")
        initial_seq_len_before_transpose = self.max_length // 2
        if initial_seq_len_before_transpose * self.base_size == 0:
            raise ValueError(f"Cannot build decoder for {self.name}: calculated dense units are zero.")
        dense_units = initial_seq_len_before_transpose * self.base_size
        x = Dense(dense_units, activation='gelu', name=f"{self.name}_dec_dense_expand")(latent_in)
        x = Reshape((initial_seq_len_before_transpose, self.base_size), name=f"{self.name}_dec_reshape")(x)
        x = Conv1DTranspose(
            self.base_size,
            kernel_size=5,
            strides=2,
            activation='gelu',
            padding='same',
            name=f"{self.name}_decoder_conv_transpose"
        )(x)
        main_logits = Conv1D(
            self.tokenizer.get_vocab_size(),
            kernel_size=1,
            activation='linear',
            padding='same',
            name=f"{self.name}_dec_to_vocab_logits"
        )(x)
        main_probs = Softmax(axis=-1, name=f"{self.name}_dec_output_softmax")(main_logits)
        main_output_named = Activation("linear", name=f"{self.name}_main_out")(main_probs)
        if self.optional:
            flag_intermediate = Dense(max(8, latent_dim // 4), activation='relu', name=f"{self.name}_dec_flag_dense")(latent_in)
            flag_intermediate = Dropout(0.1)(flag_intermediate)
            optional_flag = Dense(1, activation='sigmoid', name=f"{self.name}_dec_flag_raw")(flag_intermediate)
            flag_output_named = Activation("linear", name=f"{self.name}_flag_out")(optional_flag)
            model = tf.keras.Model(
                inputs=latent_in,
                outputs=[main_output_named, flag_output_named],
                name=f"{self.name}_decoder"
            )
            print(f"Built OPTIONAL decoder for {self.name} with outputs: {[o.name for o in model.outputs]}")
            return model
        else:
            model = tf.keras.Model(
                inputs=latent_in,
                outputs=main_output_named,
                name=f"{self.name}_decoder"
            )
            print(f"Built NON-OPTIONAL decoder for {self.name} with output: {model.output.name}")
            return model

    def print_stats(self):
        from prettytable import PrettyTable
        t = PrettyTable()
        t.field_names = [f"Text Field Stat ({self.name})", "Value"]
        t.align = "l"
        t.add_row(["Actual Vocab size", self.tokenizer.get_vocab_size() if self.tokenizer is not None else "N/A"])
        t.add_row(["PAD Token ID", self.pad_token_id])
        t.add_row(["NULL Token ID", self.null_token_id if self.optional else "N/A"])
        t.add_row(["Observed Max Tokens (no start/end)", self.dynamic_max_len])
        t.add_row(["Final Max Length (Padded)", self.max_length])
        t.add_row(["Avg Raw Length (chars)", f"{self.avg_raw_length:.2f}" if self.avg_raw_length is not None else "N/A"])
        t.add_row(["Avg Token Count", f"{self.avg_token_count:.2f}" if self.avg_token_count is not None else "N/A"])
        t.add_row(["Compression Ratio (Raw / Tokens)", f"{self.compression_ratio:.2f}" if self.compression_ratio is not None else "N/A"])
        print(t)


################################################################################
# MultiCategoryField
################################################################################

class MultiCategoryField(BaseField):
    def __init__(self, name: str, optional: bool = False):
        super().__init__(name, optional)
        self.category_set = set()
        self.category_counts = {}  # Dictionary to count the number of rows for each category.
        self.category_list = []

    def _get_input_shape(self):
        # Shape depends on finalized stats.
        return (len(self.category_list),) if self.category_list else (0,)

    def _get_output_shape(self):
        # Shape depends on finalized stats.
        return (len(self.category_list),) if self.category_list else (0,)

    def _get_loss(self):
        # Multi-label classification loss for the main output head.
        return tf.keras.losses.BinaryCrossentropy()

    def get_base_padding_value(self):
        # Padding is a vector of zeros.
        shape = self._get_output_shape()
        return tf.zeros(shape, dtype=tf.float32)

    def get_flag_padding_value(self):
         return tf.constant([1.0], dtype=tf.float32)

    def _accumulate_stats(self, raw_value):
        if raw_value:
            # Handle both single value and list of values.
            # Convert each value to a string (even if the underlying type is int) and count each unique category per row.
            cats = raw_value if isinstance(raw_value, list) else [raw_value]
            for c in set(cats):
                 cat_str = str(c)  # Ensure conversion to string.
                 if cat_str:
                     self.category_set.add(cat_str)
                     self.category_counts[cat_str] = self.category_counts.get(cat_str, 0) + 1

    def _finalize_stats(self):
        # Sort categories alphabetically first for consistent indexing
        self.category_list = sorted(list(self.category_set))
        # Calculate overall frequency (optional but potentially useful info)
        self.category_frequencies = {cat: self.category_counts.get(cat, 0) / sum(self.category_counts.values())
                                     for cat in self.category_list} if sum(self.category_counts.values()) > 0 else {}
        if not self.category_list:
             logging.warning(f"No categories found for MultiCategoryField '{self.name}'. Output shape will be (0,).")
        # Store index map for faster lookups in transform
        self.category_to_index = {cat: i for i, cat in enumerate(self.category_list)}

    def _transform(self, raw_value):
        """Transforms a non-None list/value to a multi-hot vector."""
        shape = self._get_output_shape()
        if shape[0] == 0:
            return tf.zeros(shape, dtype=tf.float32) # Handle case with no categories.

        vec = np.zeros(shape[0], dtype=np.float32)
        # Ensure input values are converted to strings.
        cats = raw_value if isinstance(raw_value, list) else ([raw_value] if raw_value is not None else [])

        for c in cats:
            c_str = str(c)
            idx = self.category_to_index.get(c_str) # Use the precomputed map
            if idx is not None:
                vec[idx] = 1.0
        return tf.constant(vec, dtype=tf.float32)

    def noise_function(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        # Optional: Add noise like randomly flipping some categories.
        # For now, no noise is added.
        return x

    def to_string(
        self,
        predicted_main: np.ndarray,
        predicted_flag: Optional[np.ndarray] = None,
        cumulative_threshold: float = 0.9, # Target cumulative probability
        min_absolute_prob: float = 0.1,   # Minimum probability for any included category
        max_categories: Optional[int] = 10 # Optional upper limit on categories shown
        ) -> str:

        if self.optional:
            if predicted_flag is None:
                raise ValueError("predicted_flag is required for optional MultiCategoryField.to_string")
            is_null_prob = float(predicted_flag[0])
            # Adjust threshold for determining null if needed, 0.5 is common
            if is_null_prob > 0.5:
                return "None"
            # If not null, proceed with processing predicted_main
            probs = predicted_main
        else:
            probs = predicted_main

        if not self.category_list or probs.size == 0: # Handle no categories or empty prediction
            return "(no categories defined)"

        # Create pairs of (index, probability)
        indexed_probs = list(enumerate(probs))

        # Sort by probability in descending order
        indexed_probs.sort(key=lambda x: x[1], reverse=True)

        selected_indices: List[int] = []
        cumulative_prob: float = 0.0

        for index, prob in indexed_probs:
            # Stop if max categories reached (if set)
            if max_categories is not None and len(selected_indices) >= max_categories:
                break

            # Stop if current prob is below absolute minimum
            # AND if we haven't selected anything yet (ensures we don't show only v. low probs)
            # OR if we have selected something, stop adding items below the minimum threshold
            if prob < min_absolute_prob:
                 # If it's the very first item considered and it's below threshold, stop entirely.
                 # Or if we've already met the cumulative goal, don't add small ones.
                 # Or just generally stop adding items below the min threshold.
                 # Let's choose the last one: stop adding any item below min_absolute_prob.
                 break # Don't add this or any subsequent categories

            # Add the category index
            selected_indices.append(index)
            cumulative_prob += prob

            # Stop if cumulative probability threshold is met
            if cumulative_prob >= cumulative_threshold:
                break

        # --- Post-selection checks ---
        # If after filtering, we still selected an item below the threshold
        # (this can happen if the *first* item was > min_abs but < target_cumul, and no others were added)
        # It might be better to enforce the min_absolute_prob strictly on *all* selected items.
        # Re-filter selected_indices based on the minimum absolute probability
        final_selected_indices = [idx for idx in selected_indices if probs[idx] >= min_absolute_prob]

        if not final_selected_indices:
            # Check if there was at least *one* category above the min threshold initially
            # If even the top one wasn't, return specific message
            if not indexed_probs or indexed_probs[0][1] < min_absolute_prob:
                 return "(all below min threshold)"
            else:
                 # This case means items were selected initially but *all* got filtered out
                 # by the final absolute check, which is less likely with the logic above,
                 # but good to handle. Could also mean cumulative target was met by items
                 # just slightly below min_absolute_prob after the first.
                 return "(none met criteria)"


        # Get category names, preserving the order of selection (most probable first)
        selected_cats = [self.category_list[i] for i in final_selected_indices]

        return ', '.join(selected_cats)
    
    def build_encoder(self, latent_dim: int) -> tf.keras.Model:
        inp = tf.keras.Input(shape=self.input_shape, name=f"{self.name}_input", dtype=self.input_dtype)
        num_categories = self._get_input_shape()[0]
        units = num_categories // 4
        x = Dense(units, activation='gelu')(inp)
        x = LayerNormalization()(x)
        x = Dense(units, activation='gelu')(x)
        out = LayerNormalization()(x)
        return tf.keras.Model(inp, out, name=f"{self.name}_encoder")

    def build_decoder(self, latent_dim: int) -> tf.keras.Model:
        inp = tf.keras.Input(shape=(latent_dim,), name=f"{self.name}_decoder_in")
        num_categories = self._get_input_shape()[0]
        units = num_categories // 4
        # Normal case with categories.
        x = Dense(units, activation='gelu')(inp)
        x = LayerNormalization()(x)
        x = Dense(units, activation='gelu')(x)
        x = LayerNormalization()(x)
        # Sigmoid activation for multi-label main output.
        main_out = Dense(self._get_output_shape()[0], activation='sigmoid', name=f"{self.name}_main_out")(x)
        return tf.keras.Model(inp, main_out, name=f"{self.name}_decoder")

    def print_stats(self):
        t = PrettyTable()
        t.field_names = [f"Category", "Row Count", "Frequency"]
        t.align = "l"
        num_cats = len(self.category_list)
        t.add_row(["Unique categories", num_cats, "100%"])

        # Sort categories by count (desc) for printing, or keep alphabetical
        # sorted_cats = sorted(self.category_list, key=lambda cat: self.category_counts.get(cat, 0), reverse=True)
        sorted_cats = self.category_list # Keep alphabetical from finalize

        for cat in sorted_cats:
            count = self.category_counts.get(cat, 0)
            freq = self.category_frequencies.get(cat, 0.0)
            t.add_row([cat, count, f"{freq:.4f}"])
        print(t)


class SingleCategoryField(BaseField):
    def __init__(self, name: str, optional: bool = False):
        super().__init__(name, optional)
        self.category_set = set()
        self.category_counts = {}  # counts how many rows contained each category
        self.category_list = []    # finalized sorted list of unique categories

    @property
    def input_dtype(self):
        return tf.int32

    @property
    def output_dtype(self):
        return tf.int32

    def _get_input_shape(self):
        return (1,)

    def _get_output_shape(self):
        return (1,)

    def _get_loss(self):
        # Use sparse categorical crossentropy; assume decoder outputs logits over vocab.
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def get_base_padding_value(self):
        return tf.constant([0], dtype=tf.int32)

    def get_flag_padding_value(self):
        return tf.constant([1.0], dtype=tf.float32)

    def _accumulate_stats(self, raw_value):
        if raw_value is not None:
            cat_str = str(raw_value)
            if cat_str:
                self.category_set.add(cat_str)
                self.category_counts[cat_str] = self.category_counts.get(cat_str, 0) + 1

    def _finalize_stats(self):
        self.category_list = sorted(list(self.category_set))
        if not self.category_list:
            logging.warning(f"No categories found for SingleCategoryField '{self.name}'.")

    def _transform(self, raw_value):
        if raw_value is None:
            if not self.optional:
                raise ValueError(f"Field '{self.name}' is not optional, but received None.")
            return self.get_base_padding_value()
        cat_str = str(raw_value)
        try:
            idx = self.category_list.index(cat_str)
        except ValueError:
            idx = 0  # Unknown gets index 0.
        return tf.constant([idx], dtype=tf.int32)

    def transform_target(self, raw_value):
        return self._transform(raw_value)

    def transform(self, raw_value):
        return self._transform(raw_value)

    def to_string(self, predicted_main: np.ndarray, predicted_flag: Optional[np.ndarray] = None) -> str:
        # Ensure we work with a 1D array of probabilities (or logits)
        vec = predicted_main.flatten() if predicted_main.ndim > 1 else predicted_main
        idx = int(np.argmax(vec))
        if idx < len(self.category_list):
            return self.category_list[idx]
        else:
            return "[Unknown]"

    def print_stats(self):
        t = PrettyTable([f"Single Category Field", self.name])
        t.add_row(["Unique categories", len(self.category_list)])
        for cat in self.category_list:
            count = self.category_counts.get(cat, 0)
            t.add_row([cat, count])
        print(t)

    def build_encoder(self, latent_dim: int) -> tf.keras.Model:
        inp = tf.keras.Input(shape=(1,), name=f"{self.name}_input", dtype=tf.int32)
        embed = tf.keras.layers.Embedding(
            input_dim=len(self.category_list),
            output_dim=len(self.category_list) // 4,
            name=f"{self.name}_embedding"
        )(inp)
        out = tf.keras.layers.Flatten()(embed)
        return tf.keras.Model(inp, out, name=f"{self.name}_encoder")

    def build_decoder(self, latent_dim: int) -> tf.keras.Model:
        inp = tf.keras.Input(shape=(latent_dim,), name=f"{self.name}_decoder_in")
        logits = tf.keras.layers.Dense(
            len(self.category_list) if self.category_list else 1,
            name=f"{self.name}_logits"
        )(inp)
        return tf.keras.Model(inp, logits, name=f"{self.name}_decoder")


class NumericDigitCategoryField(BaseField):
    def __init__(self, name: str, base: int = 10, fraction_digits: int = 0, optional: bool = False):
        super().__init__(name, optional)
        self.base = base
        self.fraction_digits = fraction_digits
        self.data_points = []
        self.has_negative = False
        self.has_nan = False
        self.integer_digits = None
        self.total_positions = None

    def _accumulate_stats(self, raw_value):
        # detect None or NaN
        if raw_value is None:
            self.has_nan = True
        else:
            try:
                val = float(raw_value)
                if math.isnan(val):
                    self.has_nan = True
                else:
                    if val < 0:
                        self.has_negative = True
                    self.data_points.append(val)
            except (ValueError, TypeError):
                # non-numeric string—treat as nan
                self.has_nan = True

    def _finalize_stats(self):
        # determine width of integer part from non-nan values
        if not self.data_points:
            self.integer_digits = 1
        else:
            abs_ints = [int(math.floor(abs(v))) for v in self.data_points]
            max_int = max(abs_ints)
            if max_int > 0:
                needed = int(math.floor(math.log(max_int, self.base))) + 1
            else:
                needed = 0
            self.integer_digits = needed or (1 if self.fraction_digits > 0 else 1)

        # compute total positions: [nan?][sign?] + integer_digits + fraction_digits
        self.total_positions = (
            (1 if self.has_nan else 0)
            + (1 if self.has_negative else 0)
            + self.integer_digits
            + self.fraction_digits
        )

    def _get_input_shape(self):
        if self.total_positions is None:
            self._finalize_stats()
        return (self.total_positions,)

    def _get_output_shape(self):
        return self._get_input_shape()

    def _get_loss(self):
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    def get_base_padding_value(self):
        return tf.constant([0] * self._get_input_shape()[0], dtype=tf.int32)

    def get_flag_padding_value(self):
        return tf.constant([1.0], dtype=tf.float32)

    def _transform(self, raw_value):
        # ensure stats finalized
        if self.total_positions is None:
            self._finalize_stats()

        # detect nan or None
        is_nan = False
        if raw_value is None:
            is_nan = True
        else:
            try:
                val = float(raw_value)
                if math.isnan(val):
                    is_nan = True
                else:
                    raw = val
            except (ValueError, TypeError):
                is_nan = True

        # if this is nan/None, set nan slot=1 and zero-fill the rest
        if is_nan:
            return tf.constant([1] + [0] * (self.total_positions - 1), dtype=tf.int32)

        seq = []
        # nan slot
        if self.has_nan:
            seq.append(0)
        # sign slot
        if self.has_negative:
            seq.append(1 if raw < 0 else 0)

        abs_val = abs(raw)
        ipart = int(math.floor(abs_val))

        # integer digits
        if self.integer_digits > 0:
            int_digits = self._int_to_digits(ipart, self.integer_digits)
        else:
            int_digits = []
        seq.extend(int_digits)

        # fraction digits
        if self.fraction_digits > 0:
            frac = abs_val - ipart
            scaled = int(round(frac * (self.base ** self.fraction_digits)))
            scaled = min(scaled, self.base**self.fraction_digits - 1)
            seq.extend(self._int_to_digits(scaled, self.fraction_digits))

        # pad if anything missing
        if len(seq) < self.total_positions:
            seq += [0] * (self.total_positions - len(seq))

        return tf.constant(seq, dtype=tf.int32)

    def to_string(self, predicted_tensor: np.ndarray, flag_tensor: Optional[np.ndarray] = None) -> str:
        """
        Converts the raw output tensor (probabilities/logits per position)
        from the decoder into a string representation of the number.
        Performs argmax internally.
        """
        # Ensure stats are finalized to get shape info
        if self.total_positions is None or self.base is None or self.integer_digits is None:
             logging.error(f"Field '{self.name}': Stats (total_positions/base/integer_digits) not finalized in to_string.")
             # Attempt to finalize, though ideally this should be done before reconstruction
             self._finalize_stats()
             if self.total_positions is None or self.base is None or self.integer_digits is None:
                 raise RuntimeError(f"Field '{self.name}': Cannot determine necessary stats even after finalize attempt.")

        expected_shape = (self.total_positions, self.base)
        if predicted_tensor.shape != expected_shape:
            # Add a check for the input shape before processing
            raise ValueError(f"Field '{self.name}': Input tensor shape {predicted_tensor.shape} does not match expected shape {expected_shape}.")

        # **** Core Change: Perform argmax here ****
        # Find the index (predicted digit) with the highest probability/logit for each position
        predicted_digits = np.argmax(predicted_tensor, axis=-1) # Shape will be (total_positions,)

        # --- The rest of the logic uses the 1D 'predicted_digits' array ---
        total_expected_digits = self.total_positions
        if len(predicted_digits) != total_expected_digits:
            # This check is unlikely to fail now but kept for sanity
            raise ValueError(f"Field '{self.name}': Length mismatch after argmax. Got {len(predicted_digits)}, expected {total_expected_digits}.")

        logging.debug(f"Field '{self.name}': Predicted digits after argmax: {predicted_digits}")

        idx = 0
        # Check for NaN representation (if applicable)
        if self.has_nan:
            if int(predicted_digits[idx]) == 1: # Assuming 1 represents NaN in the first slot
                return "NaN"
            idx += 1 # Move past the NaN slot

        # Check for sign representation (if applicable)
        negative = False
        if self.has_negative:
            negative = bool(int(predicted_digits[idx]) == 1) # Assuming 1 represents negative in the sign slot
            idx += 1 # Move past the sign slot

        # Reconstruct integer part
        int_val = 0
        num_int_digits_to_read = self.integer_digits
        try:
            int_digits_slice = predicted_digits[idx : idx + num_int_digits_to_read]
            if len(int_digits_slice) != num_int_digits_to_read:
                 logging.error(f"Field '{self.name}': Error slicing integer digits. Idx={idx}, NumRead={num_int_digits_to_read}, SliceLen={len(int_digits_slice)}, TotalDigits={len(predicted_digits)}")
                 return "[Error Slicing Int Digits]"
            for digit in int_digits_slice:
                int_val = int_val * self.base + int(digit)
            idx += num_int_digits_to_read
        except IndexError:
             logging.error(f"Field '{self.name}': IndexError reading integer digits. Idx={idx}, NumRead={num_int_digits_to_read}, TotalDigits={len(predicted_digits)}")
             return "[IndexError Reading Int Digits]"


        # Reconstruct fraction part
        frac_val = 0
        num_frac_digits_to_read = self.fraction_digits
        try:
            if num_frac_digits_to_read > 0:
                frac_digits_slice = predicted_digits[idx : idx + num_frac_digits_to_read]
                if len(frac_digits_slice) != num_frac_digits_to_read:
                    logging.error(f"Field '{self.name}': Error slicing fraction digits. Idx={idx}, NumRead={num_frac_digits_to_read}, SliceLen={len(frac_digits_slice)}, TotalDigits={len(predicted_digits)}")
                    return "[Error Slicing Frac Digits]"
                for digit in frac_digits_slice:
                    frac_val = frac_val * self.base + int(digit)
                idx += num_frac_digits_to_read
        except IndexError:
            logging.error(f"Field '{self.name}': IndexError reading fraction digits. Idx={idx}, NumRead={num_frac_digits_to_read}, TotalDigits={len(predicted_digits)}")
            return "[IndexError Reading Frac Digits]"


        # Format the final string
        s = f"{int_val}"
        if self.fraction_digits > 0:
            # Ensure the fraction part is padded with leading zeros if needed
            s += "." + f"{frac_val:0{self.fraction_digits}d}"

        # Prepend sign if necessary
        return ("-" if negative else "") + s

    def print_stats(self):
        if self.integer_digits is None or self.total_positions is None:
            self._finalize_stats()

        tbl = PrettyTable()
        tbl.field_names = [
            "Field", "Base", "Int Digits", "Frac Digits",
            "Has Sign", "Has NaN", "Total Positions", "Samples"
        ]
        tbl.add_row([
            self.name, self.base, self.integer_digits, self.fraction_digits,
            self.has_negative, self.has_nan, self.total_positions,
            len(self.data_points) + (1 if self.has_nan else 0)
        ])
        print(tbl)

    def _int_to_digits(self, value, num_digits):
        digits = []
        for _ in range(num_digits):
            digits.append(value % self.base)
            value //= self.base
        return digits[::-1]  # reverse to correct order

    def build_encoder(self, latent_dim: int) -> tf.keras.Model:
        inp = tf.keras.Input(shape=self._get_input_shape(), name=f"{self.name}_input", dtype=tf.int32)
        x = CategoryEncoding(
            num_tokens=self.base,
            output_mode="one_hot")(inp)
        x = Flatten()(x)
        total_positions = self._get_input_shape()[0]
        units = total_positions * self.base
        x = Dense(units, activation='gelu')(x)
        x = LayerNormalization()(x)
        x = Dense(units, activation='gelu')(x)
        x = LayerNormalization()(x)
        out = x
        return tf.keras.Model(inp, out, name=f"{self.name}_encoder")

    def build_decoder(self, latent_dim: int) -> tf.keras.Model:
        total_positions = self._get_input_shape()[0]
        units = total_positions * self.base
        inp = tf.keras.Input(shape=(latent_dim,), name=f"{self.name}_decoder_in")
        x = Dense(units, activation='gelu')(inp)
        x = LayerNormalization()(x)
        x = Dense(units, activation='gelu')(x)
        x = LayerNormalization()(x)
        main_logits = Dense(units, name=f"{self.name}_main_out")(x)
        main_logits = Reshape((total_positions, self.base))(main_logits)
        main_logits = Softmax(axis=-1, name=f"{self.name}_main_out_softmax")(main_logits)
        return tf.keras.Model(inp, main_logits, name=f"{self.name}_decoder")
