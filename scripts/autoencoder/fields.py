from prettytable import PrettyTable
import numpy as np
import tensorflow as tf
from enum import Enum
from tensorflow.keras.layers import (
    Embedding,
    LayerNormalization,
    Dense,
    Input,
    Reshape,
    Softmax,
    Conv1D,
    Conv1DTranspose,
    Flatten,
    Add,
    Lambda
)
from tensorflow.keras.models import Model
import abc

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

def add_positional_encoding(input_sequence):
    batch_size = tf.shape(input_sequence)[0]
    sequence_length = tf.shape(input_sequence)[1]
    pos_range_up = tf.linspace(0.0, 1.0, sequence_length)
    pos_range_up = tf.reshape(pos_range_up, (1, sequence_length, 1))
    pos_encoding_up = tf.tile(pos_range_up, [batch_size, 1, 1])

    pos_range_down = tf.linspace(1.0, 0.0, sequence_length)
    pos_range_down = tf.reshape(pos_range_down, (1, sequence_length, 1))
    pos_encoding_down = tf.tile(pos_range_down, [batch_size, 1, 1])

    return tf.concat([input_sequence, pos_encoding_up, pos_encoding_down], axis=-1)

def sin_activation(x):
    return tf.sin(x)

################################################################################
# BaseField
################################################################################

class BaseField(abc.ABC):
    def __init__(self, name: str, optional: bool = False):
        self.name = name
        self.optional = optional

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

    def accumulate_stats(self, raw_value):
        self._accumulate_stats(raw_value)

    def finalize_stats(self):
        self._finalize_stats()

    def transform(self, raw_value):
        if raw_value is None and not self.optional:
            raise ValueError(f"Received None for non-optional field: {self.name}")
        is_null = 1.0 if raw_value is None else 0.0
        base_output = self._transform(raw_value)
        if self.optional:
            return tf.concat(
                [base_output, tf.constant([is_null], dtype=tf.float32)],
                axis=-1
            )
        return base_output

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

    # New: optional helper for printing stats
    def print_stats(self):
        # Default no-op; each subclass can override.
        pass

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
        return 'binary_crossentropy' if self.use_bce_loss else 'mse'

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
        optional: bool = False
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

    def _get_input_shape(self):
        return (1,)

    def _get_output_shape(self):
        return (1,)

    def _get_loss(self):
        return 'mse'

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
            float_val = float(value_part)
        else:
            float_val = float(predicted_array)

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
        x = Dense(32, activation='gelu')(inp)
        x = LayerNormalization()(x)
        x = Dense(latent_dim // 8, activation='gelu')(x)
        out = LayerNormalization()(x)
        return tf.keras.Model(inp, out, name=f"{self.name}_encoder")

    def build_decoder(self, latent_dim: int) -> tf.keras.Model:
        inp = tf.keras.Input(shape=(latent_dim,), name=f"{self.name}_decoder_in")
        x = Dense(32, activation='gelu')(inp)
        x = LayerNormalization()(x)
        x = Dense(self.output_shape[0], activation='linear')(x)
        return tf.keras.Model(inp, x, name=f"{self.name}_decoder")

    def print_stats(self):
        t = PrettyTable([f"Scalar Field", self.name])
        t.add_row(["Count", self.n])
        t.add_row(["Min", self.min_val])
        t.add_row(["Max", self.max_val])
        t.add_row(["Mean", self.mean_val])
        t.add_row(["Std", self.std_val])
        print(t)

################################################################################
# TextField
################################################################################

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
        def masked_sparse_categorical_crossentropy(y_true, y_pred):
            pad_token_index = self.char_to_index[SPECIAL_PAD]
            mask = tf.cast(tf.not_equal(y_true, pad_token_index), dtype=tf.float32)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
            masked_loss = loss * mask
            return tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)
        return masked_sparse_categorical_crossentropy

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
                output_shape=lambda s: (s[0], s[1], s[2] + 2),
                name="add_positional_encoding")(embedding)

        current_length = self.max_length
        current_filters = self.base_size

        for step in range(self.downsample_steps):
            num_blocks = self.num_blocks_per_step[step] if step < len(self.num_blocks_per_step) else 1
            for block in range(num_blocks):
                x = self._residual_block(x, current_filters, block_num=block, step_num=step)
            current_length //= 2
            current_filters *= 2
            x = Conv1D(current_filters, 3, strides=2, activation=sin_activation, padding='same')(x)
            # x = LayerNormalization()(x)

        x = Conv1D(self.base_size, 3, activation='linear')(x)
        x = Flatten()(x)
        return tf.keras.Model(inp, x, name=f"{self.name}_encoder")

    def build_decoder(self, latent_dim: int) -> tf.keras.Model:
        inp = tf.keras.Input(shape=(latent_dim,), name=f"{self.name}_decoder_in")
        current_length = self.max_length // (2 ** self.downsample_steps)
        current_filters = self.base_size * (2 ** self.downsample_steps)

        x = Dense(current_length * self.base_size, activation=sin_activation)(inp)
        x = Reshape((current_length, self.base_size))(x)
        x = Lambda(add_positional_encoding, 
                output_shape=lambda s: (s[0], s[1], s[2] + 2),
                name="add_positional_encoding")(x)

        for step in range(self.downsample_steps):
            x = Conv1DTranspose(current_filters * 2, 3, strides=2, activation=sin_activation, padding='same')(x)
            # x = LayerNormalization()(x)
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
        x = Conv1D(filters // reduction_ratio, 1, activation=sin_activation, padding='same')(x)
        # x = LayerNormalization()(x)
        x = Conv1D(filters // reduction_ratio, kernel_size, activation=sin_activation, padding='same')(x)
        # x = LayerNormalization()(x)
        x = Conv1D(filters, 1, activation='linear', padding='same')(x)
        # x = LayerNormalization()(x)
        if shortcut.shape[-1] != x.shape[-1]:
            shortcut = Conv1D(x.shape[-1], 1, padding='same', name=f'adjustment_conv_S{step_num}_B{block_num}')(shortcut)
        return Add()([x, shortcut])

    def print_stats(self):
        t = PrettyTable([f"Text Field", self.name])
        t.add_row(["Unique chars", len(self.alphabet)])
        t.add_row(["Max raw length", self.dynamic_max_len])
        t.add_row(["Final max_length", self.max_length])
        print(t)

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
        return 'binary_crossentropy'

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
        inp = tf.keras.Input(shape=(len(self.category_list),), name=f"{self.name}_input", dtype=self.input_dtype)
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
        out = Dense(len(self.category_list), activation='sigmoid', name=f"{self.name}_decoder")(x)
        return tf.keras.Model(inp, out, name=f"{self.name}_decoder")

    def print_stats(self):
        t = PrettyTable([f"MultiCategory Field", self.name])
        t.add_row(["Unique categories", len(self.category_list)])
        for i, c in enumerate(self.category_list):
            t.add_row([i, c])
        print(t)
