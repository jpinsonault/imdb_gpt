import os
from typing import Any, Dict
import tensorflow as tf
from tensorflow.keras import layers, Model
from pathlib import Path
from config import project_config
from autoencoder.schema import RowAutoencoder
from autoencoder.imdb_row_autoencoders import TitlesAutoencoder
from scripts.autoencoder.training_callbacks import LatentCorrectorReconstructionCallback

def build_correction_network(latent_dim, hidden_units, num_layers):
    inp = tf.keras.Input(shape=(latent_dim,), name="latent_correction_input")
    x = inp
    for i in range(num_layers):
        x = layers.Dense(hidden_units, activation='gelu', name=f"siren_dense_{i}")(x)
    noise = layers.Dense(latent_dim, activation='linear', name="noise")(x)
    # Subtract the predicted noise from the noisy latent
    corrected = layers.Add(name="corrected_latent_output")([inp, -noise])
    return Model(inp, corrected, name="correction_network")

class LatentCorrector(tf.keras.Model):
    def __init__(self, base_autoencoder: RowAutoencoder, db_path: Path, config: Dict[str, Any]):
        super().__init__(name="LatentCorrector")
        self.config = config
        self.learning_rate = config["learning_rate"]
        self.epochs = config["epochs"]
        self.base_autoencoder: RowAutoencoder = base_autoencoder
        self.db_path = db_path
        self.latent_dim = base_autoencoder.latent_dim
        self.noise_std = config["noise_std"]
        # Freeze the encoder so that only the corrector (and later the decoder wrapper) is trained.
        self._encoder = base_autoencoder.encoder
        self._encoder.trainable = False

        # Build the correction network (which maps a noisy latent to a corrected latent)
        self.correction_network = build_correction_network(
            self.latent_dim, config["hidden_units"], config["num_layers"]
        )

    def call(self, inputs, training=False):
        # When calling the corrector alone it just outputs the corrected latent.
        return self.correction_network(inputs)

    def create_latent_dataset(self, base_dataset):
        def map_fn(inputs, _):
            latent = self._encoder(inputs)
            noise = tf.random.normal(shape=tf.shape(latent), mean=0.0, stddev=self.noise_std)
            noisy_latent = latent + noise
            target_reconstruction = self.base_autoencoder.decoder(latent)
            return noisy_latent, tuple(target_reconstruction)

        return base_dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    def fit_model(self):
        # First, build the training dataset from the base autoencoder's data.
        base_dataset = self.base_autoencoder._build_dataset()
        latent_dataset = self.create_latent_dataset(base_dataset)

        # Create a new model that composes the correction network and the (frozen) decoder.
        # This model takes a noisy latent as input and produces a reconstructed row.
        correction_input = tf.keras.Input(shape=(self.latent_dim,), name="noisy_latent_input")
        corrected_latent = self.correction_network(correction_input)
        # Freeze the decoder so its weights remain fixed during training.
        self.base_autoencoder.decoder.trainable = False
        decoded_outputs = self.base_autoencoder.decoder(corrected_latent)
        output_dict = {
            f"{field.name}_decoder": output 
            for field, output in zip(self.base_autoencoder.fields, decoded_outputs)
        }
        correction_decoding_model = tf.keras.Model(correction_input, output_dict, name="corrector_decoder")

        correction_decoding_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=self.base_autoencoder.get_loss_dict(),
            loss_weights=self.base_autoencoder.get_loss_weights_dict()
        )

        # (Optional) set up callbacks (you might need to adjust the reconstruction callback as well)
        log_dir = "logs/fit/" + str(int(tf.timestamp().numpy()))
        os.makedirs(log_dir, exist_ok=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            embeddings_freq=1,
            update_freq=self.config["callback_interval"]
        )
        reconstruction_callback = LatentCorrectorReconstructionCallback(
            interval_batches=self.config["callback_interval"],
            row_autoencoder=self.base_autoencoder,
            num_samples=20
        )

        # Train the composite model.
        correction_decoding_model.fit(
            latent_dataset,
            epochs=self.epochs,
            callbacks=[tensorboard_callback, reconstruction_callback]
        )

        # Optionally, save only the correction network weights.
        correction_decoding_model.save("models/latent_corrector_with_decoder.h5")

if __name__ == "__main__":
    autoencoder_config = project_config["autoencoder"]
    corrector_config = project_config["corrector"]
    model_dir = Path(project_config["model_dir"])
    data_dir = Path(project_config["data_dir"])
    db_path = data_dir / 'imdb.db'

    # Load or build your base autoencoder (for example, TitlesAutoencoder)
    base_ae = TitlesAutoencoder(autoencoder_config, db_path)
    base_ae.accumulate_stats()
    base_ae.load_model(Path("models/"))

    # Create and train the latent corrector.
    latent_corrector = LatentCorrector(base_ae, db_path, corrector_config)
    latent_corrector.fit_model()
    latent_corrector.save(str(model_dir / "latent_corrector.h5"))
