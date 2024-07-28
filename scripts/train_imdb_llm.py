from datetime import datetime
from functools import partial
import json
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
    languages = project_config['llm'].get('languages', None)
    book_ids = project_config['llm'].get('book_ids', None)
    num_books = project_config['llm'].get('num_books', None)
    book_ids = None

    alphabet = get_alphabet(languages=languages, book_ids=book_ids)
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
        loss='categorical_crossentropy',
        metrics=[NextTokenAccuracy()],
    )

    model.summary()

    gututenberg_dataset = load_gutenberg_dataset(languages=languages, book_ids=book_ids)
    tf_dataset = create_tf_dataset(gututenberg_dataset, num_books, char_to_index, input_length, batch_size)

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
        gen_length=max(50, input_length//4),
        languages=languages,
        book_ids=book_ids,
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
            (0, 0.0001),
            (1, 0.00001),
        ]
    )

    steps_per_epoch = calculate_steps_per_epoch(languages, input_length, batch_size, num_books, book_ids)

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
            cosine_lr_scheduler,
            save_model_callback,
            commit_changes_to_git_callback,
            tensorboard_callback,
            reconstruct_callback,
            keras.callbacks.TensorBoard('./logs', update_freq=5)
        ]
    )


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
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        values = tf.cast(tf.equal(y_true, y_pred), tf.float32)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            values = tf.multiply(values, sample_weight)
        self.total.assign_add(tf.reduce_sum(values))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return self.total / self.count

    def reset_state(self):
        self.total.assign(0.)
        self.count.assign(0.)

def load_gutenberg_dataset(languages=None, book_ids=None):
    """
    Load the Gutenberg dataset for specified languages and book IDs.
    
    Args:
    languages (list): List of language codes to include. If None, all languages are included.
    book_ids (list): List of book IDs to include. If None, all books are included.
    
    Returns:
    dataset: A filtered and concatenated dataset.
    """
    dataset = load_dataset("manu/project_gutenberg")
    
    if languages is None:
        languages = list(dataset.keys())
    else:
        languages = [lang for lang in languages if lang in dataset.keys()]
        if not languages:
            raise ValueError("No valid languages specified.")
    
    filtered_datasets = []
    for lang in languages:
        lang_dataset = dataset[lang]
        if book_ids:
            lang_dataset = lang_dataset.filter(lambda example: example['id'] in book_ids)
        filtered_datasets.append(lang_dataset)
    
    concatenated_dataset = concatenate_datasets(filtered_datasets)
    print(f"Loaded Gutenberg dataset for languages: {', '.join(languages)}")
    print(f"Total number of books: {len(concatenated_dataset)}")
    
    return concatenated_dataset

def preprocess_text(text):
    text = remove_gutenberg_headers_footers(text)
    text = remove_excessive_whitespace(text)
    return text

def remove_gutenberg_headers_footers(text):
    # Remove headers
    header_regex = r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK .* \*\*\*"
    text = re.split(header_regex, text, 1)[-1]
    # Remove footers
    footer_regex = r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK .* \*\*\*"
    text = re.split(footer_regex, text, 1)[0]
    return text

def remove_excessive_whitespace(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


from itertools import islice
def create_tf_dataset(dataset, num_books, char_to_index, max_input_length, batch_size):
    token_processor = TokenProcessor(char_to_index, max_input_length)

    def generate_samples():
        # take a subset of the dataset
        for book in islice(dataset, num_books):
            text = preprocess_text(book['text'])
            tokenized = token_processor.tokenize(text)
            try:
                for i in range(len(tokenized) - 2):
                    input_sequence = tokenized[i:i+max_input_length]
                    if len(input_sequence) < max_input_length:
                        input_sequence = np.pad(input_sequence, (0, max_input_length - len(input_sequence)), 
                                                mode='constant', constant_values=char_to_index[SPECIAL_PAD])
                    elif len(input_sequence) > max_input_length:
                        input_sequence = input_sequence[:max_input_length]
                    target = tokenized[i+max_input_length]
                    yield input_sequence, target
            except Exception as e:
                print(f"Error processing book: {e}")
                continue

    def generator():
        for x, y in generate_samples():
            yield x, tf.one_hot(y, depth=len(char_to_index))

    return tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.int32, tf.float32),
        output_shapes=((max_input_length,), (len(char_to_index),))
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


def create_name_generator():
    name_counts = defaultdict(int)

    def generate_unique_name(name: str) -> str:
        name_counts[name] += 1
        count = name_counts[name]
        return f"{name}{count}"

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


def create_global_query(input_sequence, character_embedding_dim):
    query_tokens = Conv1D(filters=character_embedding_dim, kernel_size=1, padding='same', activation=None, name=n('query_tokens'))(input_sequence)
    query_weights = Conv1D(filters=1, kernel_size=1, padding='same', activation='sigmoid', name=n('query_weights'))(query_tokens)
    query_weights = Softmax(axis=1)(query_weights)
    global_query = ReduceSum(axis=1, keepdims=True, name=n('global_query'))(query_tokens * query_weights)
    return global_query


def additive_self_attention(input_length, character_embedding_dim, activation, num_heads):
    def apply_attention(input_sequence):

        head_dim = character_embedding_dim // num_heads

        split_inputs = Lambda(lambda x: tf.split(x, num_heads, axis=-1))(input_sequence)

        head_outputs = []
        for i in range(num_heads):
            head_input_sequence = split_inputs[i]

            value_tokens = Conv1D(filters=head_dim, kernel_size=1, padding='same', activation=None)(head_input_sequence)
            global_query = create_global_query(head_input_sequence, head_dim)

            queried_values = value_tokens * global_query

            output_mask = Conv1D(filters=1, kernel_size=1, padding='same', activation='sigmoid')(queried_values)
            output_mask = Softmax(axis=1)(output_mask)

            queried_values = Multiply(name=n("apply_attention_output_mask"))([queried_values, output_mask])
            head_outputs.append(queried_values)

        queried_values = Concatenate(name=n("combine_heads"))(head_outputs)
        output_tokens = Conv1D(filters=character_embedding_dim, kernel_size=1, padding='same', activation=None)(queried_values)

        output_tokens = Add(name=n("add_and_norm"))([output_tokens, input_sequence])
        output_tokens = LayerNormalization()(output_tokens)
    
        return output_tokens
    
    return apply_attention


def ff_layer(character_embedding_dim, activation):
    def apply_ff_layer(input_sequence):
        x = Conv1D(filters=character_embedding_dim*4, kernel_size=1, padding='same', activation=activation)(input_sequence)
        x = Conv1D(filters=character_embedding_dim, kernel_size=1, padding='same', activation=None)(x)
        x = Add(name=n("skip_around_ff"))([x, input_sequence])
        x = LayerNormalization()(x)
        return x

    return apply_ff_layer


######################################################################################################################################
#-------------------------------------------------------------------------------------------------------------------------------------

def kermit_language_model(input_length, alphabet_size, character_embedding_dim, num_blocks=4):
    activation = 'leaky_relu'

    inputs = tf.keras.Input(shape=(input_length,), dtype=tf.int32)
    embedding_layer = Embedding(input_dim=alphabet_size, output_dim=character_embedding_dim)
    embedding = embedding_layer(inputs)

    positional_encoding = sinusoidal_encoding(input_length//4, character_embedding_dim)

    # down to 1/4 of the original length
    x = Conv1D(filters=character_embedding_dim, kernel_size=3, strides=2, padding='same', activation=None)(embedding)
    x = Conv1D(filters=character_embedding_dim, kernel_size=3, strides=2, padding='same', activation=None)(x)
    x = LayerNormalization()(x)

    most_recent_char = embedding[:, -1, :]
    most_recent_token = Reshape((1, character_embedding_dim))(x[:, -1, :])
    x *= most_recent_token

    first_residual = x

    for i in range(num_blocks):
        x = Add(name=n("in_positional_encoding"))([x, positional_encoding])

        most_recent_token_projection = Dense(character_embedding_dim, use_bias=False, activation=None)(most_recent_token)
        x = Multiply(name=n("merge_most_recent_token"))([x, most_recent_token_projection])

        x = additive_self_attention(input_length, character_embedding_dim, activation, num_heads=4)(x)
        x = ff_layer(character_embedding_dim, activation)(x)
        x = Add(name=n("out_positional_encoding"))([x, -positional_encoding])
 
    residual_conv = Conv1D(filters=character_embedding_dim, kernel_size=1, padding='same', activation=SinActivation())(first_residual)
    x += residual_conv
    x += positional_encoding

    num_reductions = 4
    reductions = []
    for i in range(num_reductions):
        most_recent_projection = Dense(character_embedding_dim, use_bias=False, activation=None)(most_recent_token)

        final_query = create_global_query(x, character_embedding_dim)
        final_query *= most_recent_projection

        final_keys = Conv1D(filters=character_embedding_dim, kernel_size=1, padding='same', activation=activation)(x)
        queried_values = Multiply(name=n('ASA_queried_values'))([final_keys, final_query])
        queried_values = Conv1D(filters=character_embedding_dim, kernel_size=1, padding='same', activation=activation, name=n('ASA_output_projection'))(queried_values)
        queried_values = LayerNormalization()(queried_values)
        weights = Conv1D(filters=1, kernel_size=1, padding='same', activation=activation)(queried_values)
        weights = Softmax(axis=1)(weights)
        queried_values = Multiply()([queried_values, weights])
        
        final_sum = ReduceSum(axis=1, keepdims=True)(queried_values)
        final_sum = Reshape((character_embedding_dim,))(final_sum)
        reduction = Dense(character_embedding_dim, use_bias=False, activation=None)(final_sum)
        reductions.append(reduction)

    residual = Add()(reductions)
    reductions = Concatenate()(reductions)

    x = Dense(character_embedding_dim*8, activation=activation)(Flatten()(reductions))
    x = LayerNormalization()(x)
    x = Dense(character_embedding_dim, activation=None)(x)
    x += residual

    x += most_recent_char

    x = Lambda(lambda x: tf.matmul(x, embedding_layer.embeddings, transpose_b=True))(x)

    x = Softmax()(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


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
    def __init__(self, char_to_index, index_to_char, max_input_length, num_samples, frequency, languages=None, book_ids=None, temperature=1.0, gen_length=25):
        self.char_to_index = char_to_index
        self.index_to_char = index_to_char
        self.max_input_length = max_input_length
        self.num_samples = num_samples
        self.frequency = frequency
        self.dataset = load_gutenberg_dataset(languages, book_ids)
        self.temperature = temperature
        self.gen_length = gen_length

    def on_epoch_end(self, epoch, logs):
        print(f"\nEpoch {epoch + 1} Autoencoder Outputs:\n")
        self._print_predictions()

    def on_batch_end(self, batch, logs=None):
        if batch % self.frequency == 0:
            print(f"\nBatch {batch} Autoencoder Outputs:\n")
            self._print_predictions()

    def _get_random_sequence(self):
        random_index = np.random.randint(0, len(self.dataset))
        example = self.dataset[random_index]
        text = example['text']
        if len(text) > self.max_input_length:
            start = np.random.randint(0, len(text) - self.max_input_length)
            return text[start:start+self.max_input_length]
        return text.ljust(self.max_input_length)  # Pad if shorter than max_input_length

    def _tokenize(self, sequence):
        return np.array([self.char_to_index.get(char, self.char_to_index[SPECIAL_PAD]) for char in sequence])

    def _select_top_character(self, predictions):
        return np.argmax(predictions)

    def _print_predictions(self):
        for _ in range(self.num_samples):
            input_sequence = self._get_random_sequence()
            input_tensor = self._tokenize(input_sequence)
            
            generated_sequence = list(input_sequence)
            color_coded_sequence = []
            
            for i in range(self.gen_length):
                try:
                    input_tensor_reshaped = np.reshape(input_tensor, (1, self.max_input_length))
                    prediction = self.model.predict(input_tensor_reshaped, verbose=0)
                    prediction = prediction[0]

                    if i == 0:
                        top_5_indices = np.argsort(prediction)[-5:][::-1]
                        top_5_chars = [self.index_to_char[idx] for idx in top_5_indices]
                        top_5_probs = prediction[top_5_indices]
                        print(f"Step {i} - Top 5 predictions:")
                        for char, prob in zip(top_5_chars, top_5_probs):
                            bar_length = 40
                            bar_fill = int(prob * bar_length)
                            bar = '#' * bar_fill + '-' * (bar_length - bar_fill)
                            print(f"{char}: {prob:.4f} | {bar}")

                    predicted_index = self._select_top_character(prediction)
                    predicted_char = self.index_to_char[predicted_index]
                    confidence = prediction[predicted_index]
                    
                    color_coded_char = self._color_code(predicted_char, confidence)
                    color_coded_sequence.append(color_coded_char)
                    
                    # Append the predicted character to the generated sequence
                    generated_sequence.append(predicted_char)
                    
                    # Update input_tensor for the next iteration
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



def calculate_steps_per_epoch(languages, max_input_length, batch_size, num_books, book_ids=None, cache_file='dataset_size_cache.json'):
    # Check if we have a cached size
    cache_key = f"{','.join(sorted(languages or []))}-{','.join(sorted(book_ids or []))}-{num_books}"
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
            if cache_key in cached_data:
                total_windows = cached_data[cache_key]
                print(f"Using cached total windows: {total_windows}")
                return total_windows // batch_size

    # Load the Gutenberg dataset
    dataset = load_gutenberg_dataset(languages, book_ids)

    total_windows = 0

    for book in tqdm(islice(dataset, num_books), desc="Calculating windows", unit="book"):
        text_length = len(book['text'])
        # Calculate the number of windows in this book
        book_windows = max(0, (text_length - max_input_length) + 1)
        total_windows += book_windows

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

    print(f"Total windows across all books: {total_windows}")
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {steps_per_epoch}")

    return steps_per_epoch



def get_alphabet(cache_file: str = 'overall_unique_chars.json', languages=None, book_ids=None) -> Set[str]:
    cache_key = f"{','.join(sorted(languages or []))}-{','.join(sorted(book_ids or []))}"
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
            if cache_key in cached_data:
                alphabet = set(cached_data[cache_key])
                print(f"Using cached alphabet for {cache_key}")
                alphabet.update([SPECIAL_PAD, SPECIAL_START, SPECIAL_END])
                return alphabet

    dataset = load_gutenberg_dataset(languages, book_ids)
    
    alphabet = set()
    for example in tqdm(dataset, desc="Processing dataset", unit="book"):
        alphabet.update(set(example["text"]))

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
                print(f"Setting learning rate to {new_learning_rate}")
                break


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    main(args)
