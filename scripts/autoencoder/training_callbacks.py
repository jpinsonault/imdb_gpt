import random
import numpy as np
from prettytable import PrettyTable, TableStyle
from typing import List, Dict
import tensorflow as tf
from .schema import RowAutoencoder
from .fields import BaseField

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
        table.field_names = ["Field", "Original", "No-Correct Recons", "Corrected Recons"]
        table.align = "l"

        # Process each sampled row
        for i, row_dict in enumerate(self.samples):
            print(f"\nSample {i + 1}:")
            # Transform the raw row into input tensors (one per field)
            input_tensors = {
                field.name: field.transform(row_dict.get(field.name))
                for field in self.row_autoencoder.fields
            }
            print(f"input_tensors: {input_tensors}")
            # Build a list of inputs in the order of the fields.
            # Convert to NumPy arrays and add the batch dimension.
            inputs = [
                np.expand_dims(
                    x.numpy() if hasattr(x, "numpy") else x,
                    axis=0
                )
                for field, x in sorted(input_tensors.items(), key=lambda item: item[0])
            ]

            for input in zip(self.row_autoencoder.fields, inputs):
                print(f"input: {input[0].name}, len(input[1]): {len(input[1])}")
                print(f"input shape: {input[1].shape}")



            # Get the latent vector from the base encoder
            latent = self.row_autoencoder.encoder.predict(inputs)
            # Add noise
            noise = np.random.normal(loc=0.0, scale=self.noise_std, size=latent.shape)
            noisy_latent = latent + noise

            # Decode the noisy latent directly (without correction)
            preds_no_correct = self.row_autoencoder.decoder.predict(noisy_latent)
            # Correct the noisy latent using the latent corrector (self.model)
            corrected_latent = self.model.predict(noisy_latent)
            # Decode the corrected latent vector
            preds_corrected = self.row_autoencoder.decoder.predict(corrected_latent)

            # Ensure predictions are lists (one per field) even if there is a single output
            if not isinstance(preds_no_correct, list):
                preds_no_correct = [preds_no_correct]
            if not isinstance(preds_corrected, list):
                preds_corrected = [preds_corrected]

            # For each field, extract and convert predictions
            for idx, field in enumerate(self.row_autoencoder.fields):
                # The original (raw) value for reference
                original_value = row_dict.get(field.name, "N/A")
                if isinstance(original_value, list):
                    original_value = ", ".join(original_value)
                # Take the first (and only) example from the batch dimension.
                pred_no_correct = preds_no_correct[idx][0]
                pred_corrected = preds_corrected[idx][0]
                # Use the field’s to_string helper to obtain a readable version.
                recon_no_correct = field.to_string(pred_no_correct)
                recon_corrected = field.to_string(pred_corrected)
                table.add_row([field.name, str(original_value), recon_no_correct, recon_corrected])
            print(table)
            table.clear_rows()
