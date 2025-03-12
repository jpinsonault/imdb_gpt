import random
import numpy as np
from prettytable import PrettyTable, TableStyle
from typing import List, Dict
import tensorflow as tf
from .row_autoencoder import RowAutoencoder
from .fields import BaseField
import os

class ModelSaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, row_autoencoder: RowAutoencoder, output_dir):
        super().__init__()
        self.row_autoencoder = row_autoencoder
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        self.row_autoencoder.save_model()
        print(f"Model saved to {self.output_dir} at the end of epoch {epoch+1}")


class ReconstructionCallback(tf.keras.callbacks.Callback):
    def __init__(self, interval_batches, row_autoencoder: RowAutoencoder, db_path, num_samples=5):
        super().__init__()
        self.interval_batches = interval_batches
        self.row_autoencoder = row_autoencoder
        self.db_path = db_path
        self.num_samples = num_samples

        all_rows = []
        for idx, row_dict in enumerate(row_autoencoder.row_generator()):
            all_rows.append(row_dict)
        self.samples = random.sample(all_rows, min(num_samples, len(all_rows)))

    def on_train_batch_end(self, batch, logs=None):
        if (batch + 1) % self.interval_batches != 0:
            return

        print(f"\nBatch {batch + 1}: Reconstruction Results")
        
        table = PrettyTable()
        table.set_style(TableStyle.SINGLE_BORDER)

        table.field_names = ["Field", "Original Value", "Reconstructed Value"]
        
        # Set alignment for all columns to left-justified
        table.align = "l"

        for i, row_dict in enumerate(self.samples):
            print(f"\nSample {i + 1}:")
            reconstructed_row = {}

            input_tensors = {
                field.name: field.transform(row_dict.get(field.name))
                for field in self.row_autoencoder.fields
            }

            inputs = [input_tensors[field.name] for field in self.row_autoencoder.fields]
            predictions = self.model.predict([np.expand_dims(x, axis=0) for x in inputs])

            # Ensure predictions is a list
            if not isinstance(predictions, list):
                predictions = [predictions]

            for idx, field in enumerate(self.row_autoencoder.fields):
                field_prediction = predictions[idx]

                # Check the shape of field_prediction[0]
                if isinstance(field_prediction[0], np.int64):
                    print(f"Unexpected scalar prediction for field '{field.name}'. Skipping.")
                    reconstructed_str = "Error"
                else:
                    reconstructed_str = field.to_string(field_prediction[0])
                
                reconstructed_row[field.name] = reconstructed_str
                original_value = row_dict.get(field.name, "N/A")
                if isinstance(original_value, list):
                    original_value = ", ".join(original_value)
                table.add_row([
                    field.name,
                    original_value,
                    reconstructed_str
                ])


            print(table)
            table.clear_rows()


class LatentCorrectorReconstructionCallback(tf.keras.callbacks.Callback):
    """
    This callback shows reconstruction results from the latent corrector.
    For each sample row:
      1. Transform the row using the row autoencoder’s fields.
      2. Use the base encoder to obtain a latent vector.
      3. Add noise to the latent vector.
      4. Decode the noisy latent directly (no correction).
      5. Feed the noisy latent through the corrector (i.e. self.model)
         and then decode the corrected latent.
      6. Display a table comparing the original field value,
         the reconstruction from the noisy latent, and the reconstruction
         after correction.
    """
    def __init__(self, interval_batches, row_autoencoder, noise_std=0.1, num_samples=5):
        super().__init__()
        self.interval_batches = interval_batches
        self.row_autoencoder = row_autoencoder
        self.noise_std = noise_std
        self.num_samples = num_samples

        # Accumulate samples from the row generator
        all_rows = list(self.row_autoencoder.row_generator())
        self.samples = random.sample(all_rows, min(num_samples, len(all_rows)))

    def on_train_batch_end(self, batch, logs=None):
        if (batch + 1) % self.interval_batches != 0:
            return

        print(f"\nBatch {batch + 1}: Latent Correction Reconstruction Results")
        table = PrettyTable()
        table.set_style(TableStyle.SINGLE_BORDER)
        table.field_names = ["Field", "Truth", "Autoencoded", "With Noise", "Corrected"]
        table.align = "l"

        # Process each sampled row
        for i, row_dict in enumerate(self.samples):
            print(f"\nSample {i + 1}:")
            # Transform the raw row into input tensors (one per field)
            input_tensors = {
                field.name: field.transform(row_dict.get(field.name))
                for field in self.row_autoencoder.fields
            }
            # Build a list of inputs in the order of the fields.
            # Convert to NumPy arrays and add the batch dimension.
            inputs = [
                np.expand_dims(
                    input_tensors[field.name].numpy() if hasattr(input_tensors[field.name], "numpy") else input_tensors[field.name],
                    axis=0
                )
                for field in self.row_autoencoder.fields
            ]
            
            combined_inputs = tf.concat(inputs, axis=0)

            # Get the latent vector from the base encoder
            latent = self.row_autoencoder.encoder.predict(combined_inputs)
            # Add noise
            noise = np.random.normal(loc=0.0, scale=self.noise_std, size=latent.shape)
            noisy_latent = latent + noise

            autoencoded_decodings = self.row_autoencoder.decoder.predict(latent)

            # Decode the noisy latent directly (without correction)
            noisy_decodings = self.row_autoencoder.decoder.predict(noisy_latent)
            # Correct the noisy latent using the latent corrector (self.model)
            corrected_decodings = self.model.predict(noisy_latent)
            # Decode the corrected latent vector
            corrected_decodings = self.row_autoencoder.decoder.predict(corrected_decodings)

            # Ensure predictions are lists (one per field) even if there is a single output
            if not isinstance(noisy_decodings, list):
                noisy_decodings = [noisy_decodings]
            if not isinstance(corrected_decodings, list):
                corrected_decodings = [corrected_decodings]

            # For each field, extract and convert predictions
            for idx, field in enumerate(self.row_autoencoder.fields):
                # The original (raw) value for reference
                original_value = row_dict.get(field.name, "N/A")
                if isinstance(original_value, list):
                    original_value = ", ".join(original_value)
                # Take the first (and only) example from the batch dimension.
                noisy_decoding = noisy_decodings[idx][0]
                corrected_decoding = corrected_decodings[idx][0]
                # Use the field’s to_string helper to obtain a readable version.
                noisy_output = field.to_string(noisy_decoding)
                corrected_output = field.to_string(corrected_decoding)
                autoencoded_output = field.to_string(autoencoded_decodings[idx][0])
                table.add_row([field.name, str(original_value), autoencoded_output, noisy_output, corrected_output])
            print(table)
            table.clear_rows()
