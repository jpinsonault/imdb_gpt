import random
import numpy as np
from prettytable import PrettyTable
from typing import List, Dict
import tensorflow as tf
from .schema import RowAutoencoder
from .fields import BaseField

class ReconstructionCallback(tf.keras.callbacks.Callback):
    def __init__(self, interval_batches, row_autoencoder, db_path, num_samples=5):
        super().__init__()
        self.interval_batches = interval_batches
        self.row_autoencoder = row_autoencoder
        self.db_path = db_path
        self.num_samples = num_samples

        # Grab up to 5000 rows, sample 'num_samples' from them for reconstructions
        all_rows = []
        for idx, row_dict in enumerate(row_autoencoder.row_generator(db_path)):
            if idx >= 50000:
                break
            all_rows.append(row_dict)
        self.samples = random.sample(all_rows, min(num_samples, len(all_rows)))

    def on_train_batch_end(self, batch, logs=None):
        if (batch + 1) % self.interval_batches != 0:
            return

        print(f"\nBatch {batch + 1}: Reconstruction Results")
        
        table = PrettyTable()
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
                    original_value = ",".join(original_value)
                table.add_row([
                    field.name,
                    original_value,
                    reconstructed_str
                ])

            print(table)
            table.clear_rows()