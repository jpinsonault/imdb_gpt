from datetime import datetime
import json
import sys
import traceback
from typing import List
# import opencv
from tqdm import tqdm
from pathlib import Path
import numpy as np
from collections import defaultdict
import random
import tensorflow as tf
from tensorflow import keras
import re
from tensorflow.keras import layers
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention, LayerNormalization, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, UpSampling1D, Add, Reshape, BatchNormalization, Layer, Conv1DTranspose, LeakyReLU, DepthwiseConv1D, Conv1DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from config import project_config

tf_version = tf.__version__

SPECIAL_PAD = '\u200C'

class TokenProcessor:
    def __init__(self, char_to_index, max_input_length, mask_percentage):
        self.char_to_index = char_to_index
        self.index_to_char = {index: char for char, index in char_to_index.items()}
        self.index_to_char[char_to_index[SPECIAL_PAD]] = '@'
        self.max_input_length = max_input_length
        self.mask_percentage = mask_percentage

    def tokenize(self, input_string):
        input_indices = [self.char_to_index[char] for char in input_string]
       
        # Pad the input to max_input_length using zeros
        total_pad = self.max_input_length - len(input_indices)
        num_to_pad_left = random.randint(0, total_pad)
        num_to_pad_right = total_pad - num_to_pad_left
       
        input_indices = [self.char_to_index[SPECIAL_PAD]]*num_to_pad_left + input_indices + [self.char_to_index[SPECIAL_PAD]]*num_to_pad_right
        return np.array(input_indices)

    def indices_to_string(self, indices):
        return "".join([self.index_to_char[index] for index in indices])
   
    def tokenize_and_mask_input(self, input_string):
        input_tensor = self.tokenize(input_string)
        return input_tensor
   

def main(args):
    character_autoencoder_training()


def character_autoencoder_training():
    data_dir                = Path(project_config['data_dir'])
    corpus_file_path        = data_dir / 'entities.txt'
    corpus_stats_file_path  = data_dir / 'corpus_stats.json'
    max_input_length        = project_config['entities']['max_entity_length']
    batch_size              = project_config['search_autoencoder']['batch_size']
    character_embedding_dim = project_config['search_autoencoder']['character_embedding_dim']
    latent_dim              = project_config['search_autoencoder']['latent_dim']

    save_corpus_stats(corpus_file_path)

    print_corpus_stats(corpus_stats_file_path)

    alphabet, counts = get_corpus_stats(corpus_stats_file_path)
    alphabet = [SPECIAL_PAD] + alphabet

    char_to_index = {char: index for index, char in enumerate(alphabet)}
    index_to_char = {index: char for index, char in enumerate(alphabet)}


    num_blocks_per_resolution = 2
    

    fresh_model = kermit_autoencoder(input_length=max_input_length,
                                     alphabet_size=len(alphabet),
                                     character_embedding_dim=character_embedding_dim,
                                     latent_dim=latent_dim,
                                     num_blocks_per_resolution=num_blocks_per_resolution)
   

    model_path = "models/imdb_autoencoder.h5"
    model_path = data_dir / model_path
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
        loss=masked_categorical_crossentropy,
        metrics=["accuracy"],
    )

    model.summary()

    masked_percentage = 0.15
    token_processor = TokenProcessor(
        char_to_index=char_to_index,
        max_input_length=max_input_length,
        mask_percentage=masked_percentage,
    )

    dataset = tf.data.Dataset.from_generator(lambda: autoencoder_batch_generator(corpus_file_path, batch_size, token_processor),
                                             output_types=(tf.int32, tf.float32),
                                             output_shapes=((batch_size, max_input_length, ),
                                                            (batch_size, max_input_length, len(char_to_index))))

    save_frequency = 50

    save_model_by_batch_callback = keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        save_best_only=False,
        monitor="loss",
        save_freq=save_frequency,
        verbose=1
    )

    save_model_callback = keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        save_best_only=False,
        monitor="loss",
        verbose=1
    )

    logs_dir = Path("logs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    logs_dir.mkdir(parents=True, exist_ok=True)

    reconstruct_callback = ReconstructCallback(corpus_file_path, token_processor, num_samples=10, frequency=save_frequency)

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=str(logs_dir), histogram_freq=1)

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
        save_model_by_batch_callback,
        tensorboard_callback
    ])


def gelu_activation(x):
    return tf.keras.activations.gelu(x, approximate=True)

def kermit_autoencoder(input_length, alphabet_size, character_embedding_dim, latent_dim, num_blocks_per_resolution):
    leaky_relu = lambda x: tf.nn.leaky_relu(x, alpha=0.01)
    
    gelu = lambda x: tf.keras.activations.gelu(x, approximate=True)

    inputs = Input(shape=(input_length,), dtype=tf.int32)
    embedding = Embedding(input_dim=alphabet_size, output_dim=character_embedding_dim, name="embedding")(inputs)

    x = embedding
    # Initial blocks at full resolution, no downsampling
    for _ in range(num_blocks_per_resolution - 1):
        # depthwise separable convolution
        residual = x
        conv_block_full = DepthwiseConv1D(kernel_size=3, strides=1, padding="same", activation='gelu')
        x = conv_block_full(x)
        x = Add()([x, residual])
        x = LayerNormalization()(x)

    conv_half_expand_dims = Conv1D(filters=character_embedding_dim*2, kernel_size=1, strides=1, padding="same", activation='gelu')
    x = conv_half_expand_dims(x)
    x = LayerNormalization()(x)
    
    # Last block at full resolution with downsampling
    conv_block_full_downsample = DepthwiseConv1D(kernel_size=3, strides=2, padding="same", activation='gelu')
    x = conv_block_full_downsample(x)
    x = LayerNormalization()(x)

    # Intermediate blocks at half resolution, no downsampling
    for _ in range(num_blocks_per_resolution - 1):
        residual = x
        conv_block_half = DepthwiseConv1D(kernel_size=3, strides=1, padding="same", activation='gelu')
        x = conv_block_half(x)
        x = Add()([x, residual])
        x = LayerNormalization()(x)

    conv_quarter_expand_dims = Conv1D(filters=character_embedding_dim*4, kernel_size=1, strides=1, padding="same", activation='gelu')
    x = conv_quarter_expand_dims(x)
    x = LayerNormalization()(x)
    
    # Last block at half resolution with downsampling
    conv_block_half_downsample = DepthwiseConv1D(kernel_size=3, strides=2, padding="same", activation='gelu')
    x = conv_block_half_downsample(x)
    x = LayerNormalization()(x)

    # Final blocks at quarter resolution
    for _ in range(num_blocks_per_resolution):
        residual = x
        conv_block_quarter = DepthwiseConv1D(kernel_size=3, strides=1, padding="same", activation='gelu')
        x = conv_block_quarter(x)
        x = Add()([x, residual])
        x = LayerNormalization()(x)

    # Bottleneck
    bottleneck = Flatten()(x)
    bottleneck = Dense(latent_dim, activation=None)(bottleneck)
    residual = bottleneck
    # bottleneck = Dense(latent_dim*2, activation='gelu')(bottleneck)
    bottleneck = Dense(latent_dim, activation='gelu')(bottleneck)
    bottleneck = Add()([bottleneck, residual])
    bottleneck = LayerNormalization()(bottleneck)

    # Start Decoder
    back_to_quarter = Dense(character_embedding_dim*4 * (input_length//4), activation='gelu')(bottleneck)
    back_to_quarter = Reshape((input_length//4, character_embedding_dim*4))(back_to_quarter)

    # Upsampling and applying ResNeXt blocks at quarter resolution
    for _ in range(num_blocks_per_resolution):
        residual = back_to_quarter
        conv_block_back_up_to_quarter = Conv1DTranspose(filters=character_embedding_dim*4, kernel_size=3, strides=1, padding="same", activation='gelu')
        x = conv_block_back_up_to_quarter(back_to_quarter)
        x = Add()([x, residual])
        x = LayerNormalization()(x)
        
    conv_back_up_to_half_reduce_dims = Conv1DTranspose(filters=character_embedding_dim*2, kernel_size=3, strides=2, padding="same", activation='gelu')
    x = conv_back_up_to_half_reduce_dims(x)
    x = LayerNormalization()(x)

    # Applying conv blocks at half resolution
    for _ in range(num_blocks_per_resolution):
        residual = x
        conv_block_back_up_to_half = Conv1DTranspose(filters=character_embedding_dim*2, kernel_size=3, strides=1, padding="same", activation='gelu')
        x = conv_block_back_up_to_half(x)
        x = Add()([x, residual])
        x = LayerNormalization()(x)
        
    conv_back_up_to_full_reduce_dims = Conv1DTranspose(filters=character_embedding_dim, kernel_size=1, strides=2, padding="same", activation='gelu')
    x = conv_back_up_to_full_reduce_dims(x)
    x = LayerNormalization()(x)

    # Applying conv blocks at full resolution
    for _ in range(num_blocks_per_resolution):
        residual = x
        conv_block_back_up_to_full = Conv1DTranspose(filters=character_embedding_dim, kernel_size=3, strides=1, padding="same", activation='gelu')
        x = conv_block_back_up_to_full(x)
        x = Add()([x, residual])
        x = LayerNormalization()(x)
        
    # Final 1x1 convolution to map back to alphabet size
    decoder_1x1 = Conv1D(filters=alphabet_size, kernel_size=1, strides=1, padding="same", activation="softmax")
    x = decoder_1x1(x)

    model = Model(inputs=inputs, outputs=x, name="kermit_autoencoder")
    return model


def autoencoder_batch_generator(corpus_file_path, batch_size, token_processor: TokenProcessor):
    x_batch = np.zeros((batch_size, token_processor.max_input_length), dtype=np.int32)
    y_batch = np.zeros((batch_size, token_processor.max_input_length, len(token_processor.char_to_index)), dtype=np.float32)  

    batch_index = 0

    with open(corpus_file_path, 'r') as file:
        for line in file:
            line = line.strip()
           
            # Skip empty lines
            if not line:
                continue
            
            try:
                # Process each line as a separate training example
                total_pad = token_processor.max_input_length - len(line)
                num_to_pad_left = random.randint(0, total_pad)
                num_to_pad_right = total_pad - num_to_pad_left
                line = SPECIAL_PAD * num_to_pad_left + line + SPECIAL_PAD * num_to_pad_right

                input_string_token_indices = token_processor.tokenize(line)[0]

                x_batch[batch_index] = input_string_token_indices
                for idx, token_index in enumerate(input_string_token_indices):
                    y_batch[batch_index, idx, token_index] = 1

                batch_index += 1

                if batch_index == batch_size:
                    batch_index = 0
                    yield x_batch, y_batch
                    x_batch = np.zeros((batch_size, token_processor.max_input_length), dtype=np.int32)
                    y_batch = np.zeros((batch_size, token_processor.max_input_length, len(token_processor.char_to_index)), dtype=np.float32)
            except Exception as e:
                # print the exception
                traceback.print_exc()
                exc_type, exc_obj, exc_tb = sys.exc_info()
                print(f"{exc_type} {exc_obj} {exc_tb.tb_lineno}")

                print(f"Failed to process line: {line}")
                continue

    # Yield remaining batch if it's not empty
    # if batch_index > 0:
        # yield x_batch[:batch_index], y_batch[:batch_index]

def masked_categorical_crossentropy(y_true, y_pred):
    # Identify padding tokens (which should have reduced weight)
    padding_mask = K.cast(K.equal(K.argmax(y_true, axis=-1), char_to_index[SPECIAL_PAD]), K.floatx())

    mask = 1.0 - 0.99 * padding_mask

    # Calculate categorical crossentropy
    loss = keras.losses.categorical_crossentropy(y_true, y_pred)

    # Apply the weighted mask to the loss
    weighted_loss = loss * mask

    # Calculate mean loss, considering only the non-zero weights in the mask
    return K.sum(weighted_loss) / K.sum(mask)


class ReconstructCallback(keras.callbacks.Callback):
    def __init__(self, corpus_file_path, token_processor: TokenProcessor, num_samples, frequency):
        self.corpus_file_path = corpus_file_path
        self.token_processor = token_processor
        self.num_samples = num_samples
        self.frequency = frequency

    def on_batch_end(self, epoch, logs=None):
        if epoch % self.frequency != 0:
            return
       
        print(f"\nReconstruction examples at the end of epoch {epoch}:")
        with open(self.corpus_file_path, 'r') as file:
            lines = file.readlines()

        selected_lines = random.sample(lines, self.num_samples)
        for line in selected_lines:
            original = line.strip()
            tokenized = self.token_processor.tokenize(original)
            prediction = self.model.predict(tokenized, verbose=0)
            reconstructed = self.token_processor.indices_to_string(np.argmax(prediction, axis=-1)[0])

            original_restored = self.token_processor.indices_to_string(tokenized[0])

            print(f"Original      | {original_restored}")
            print(f"Reconstructed | {reconstructed}\n")


def are_models_same(model1, model2):
    """Compare two models internal shapes"""
    model1_shapes = [layer.shape for layer in model1.get_weights()]
    model2_shapes = [layer.shape for layer in model2.get_weights()]

    return model1_shapes == model2_shapes


def try_load_model(model_path: Path):
    model_path.mkdir(parents=True, exist_ok=True)
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


def print_corpus_stats(filename):
    """Print the stats of the corpus"""
    alphabet, counts = get_corpus_stats(filename)

    most_common = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20]
    least_common = sorted(counts.items(), key=lambda x: x[1])[:20]

    print(f"Most common characters: {most_common}")
    print(f"Least common characters: {least_common}")

    print(f"Total unique characters: {len(alphabet)}")


class SlowDecayLearningRateCallback(keras.callbacks.Callback):
    """Slowly decay the learning rate"""

    def __init__(self, initial_learning_rate, min_learning_rate, decay_steps):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.min_learning_rate = min_learning_rate
        self.decay_rate = self.min_learning_rate / self.initial_learning_rate
        self.decay_steps = decay_steps

    def on_epoch_begin(self, epoch, logs=None):
        # decay the learning rate
        new_learning_rate = self.initial_learning_rate * \
            (self.decay_rate ** (epoch / self.decay_steps))
        self.model.optimizer.learning_rate.assign(new_learning_rate)
        print(f"setting learning rate to {new_learning_rate}")


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


def indices_to_string(indices, index_to_char):
    """Convert a numpy array of indices to a string"""
    indices = np.array(indices).reshape(-1).astype(int)
    return "".join([index_to_char[index] for index in indices])


def predict_string(model, input_string, num_iterations, char_to_index, index_to_char, max_input_length):
    """Predict the next character in the string"""
    # pad the input string to max_input_length
    num_to_pad = max_input_length - len(input_string)
    input_string = SPECIAL_PAD * num_to_pad + input_string

    # efficiently make the string into an array of indices. Create a numpy mapping out of the python dict char_to_index
    input_string_indices = np.array(
        [char_to_index[char] for char in input_string])
    input_string_indices = input_string_indices.reshape((1, max_input_length,))

    output_string = input_string

    # loop through the model and predict the next character
    for _ in tqdm(range(num_iterations)):
        predictions = model.predict(input_string_indices, verbose=0)[0]

        next_char_index = np.argmax(predictions)

        next_char = index_to_char[next_char_index]

        output_string += next_char

        # remove the first character from the input string and add the next character
        input_string_indices = input_string_indices[:, 1:]
        input_string_indices = np.append(
            input_string_indices, [[next_char_index]], axis=1)

    return output_string


def get_corpus_stats(corpus_stats_filename):
    """Load the corpus stats from the json file"""
    with open(corpus_stats_filename, "r", encoding="utf-8") as file:
        corpus_stats = json.load(file)
    return corpus_stats["alphabet"], corpus_stats["character_counts"]


def save_corpus_stats(corpus_file):
    """Create a json file that describes the alphabet and character counts for the corpus"""
    character_counts = defaultdict(int)
    
    print(f"Analyzing corpus file {corpus_file}")

    with open(corpus_file, "r", encoding="utf-8") as file:
        for line in tqdm(file, desc="Counting characters"):
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            for char in line:
                character_counts[char] += 1
                
    all_chars = sorted(character_counts.keys())

    # save json file
    with open("corpus_stats.json", "w") as f:
        stats = {
            "alphabet": all_chars,
            "character_counts": character_counts
        }

        json.dump(stats, f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    main(args)
