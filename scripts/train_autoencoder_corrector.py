import os
from typing import Any, Dict
import tensorflow as tf
from tensorflow.keras import layers, Model
from pathlib import Path
from config import project_config
from autoencoder.schema import RowAutoencoder
from autoencoder.imdb_row_autoencoders import TitlesAutoencoder
from scripts.autoencoder.training_callbacks import LatentCorrectorReconstructionCallback

def build_correction_network(latent_dim, hidden_units=256, num_layers=3):
    inp = tf.keras.Input(shape=(latent_dim,), name="latent_correction_input")
    x = inp
    for i in range(num_layers):
        x = layers.Dense(hidden_units, activation=tf.sin, name=f"siren_dense_{i}")(x)
    correction = layers.Dense(latent_dim, activation='linear', name="corrected_latent")(x)
    corrected = layers.Add(name="corrected_latent_output")([inp, correction])
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
        # Store the encoder as a private attribute so it isn’t traversed by summary.
        self._encoder = base_autoencoder.encoder
        self._encoder.trainable = False

        self.correction_network = build_correction_network(latent_dim, config["hidden_units"], config["num_layers"])

    def call(self, inputs, training=False):
        return self.correction_network(inputs)

    def create_latent_dataset(self, base_dataset, noise_std=None):
        if noise_std is None:
            noise_std = self.noise_std

        def map_fn(inputs, _):
            latent = self._encoder(inputs)
            noise = tf.random.normal(shape=tf.shape(latent), mean=0.0, stddev=noise_std)
            return latent + noise, latent

        return base_dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # Override summary to show only the correction network
    def summary(self, *args, **kwargs):
        self.correction_network.summary(*args, **kwargs)

    def fit_model(self):
        latent_corrector.build((None, self.latent_dim))
        latent_corrector.summary()
        
        base_dataset = self.base_autoencoder._build_dataset()
        latent_dataset = self.create_latent_dataset(base_dataset)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.compile(optimizer=optimizer, loss="mse")

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

        self.fit(latent_dataset, epochs=self.epochs, callbacks=[tensorboard_callback, reconstruction_callback])

if __name__ == "__main__":
    autoencoder_config = project_config["autoencoder"]
    corrector_config = project_config["corrector"]
    model_dir = Path(project_config["model_dir"])
    latent_dim = autoencoder_config["latent_dim"]
    data_dir = Path(project_config["data_dir"])
    db_path = data_dir / 'imdb.db'

    base_ae = TitlesAutoencoder(autoencoder_config, db_path)
    base_ae.accumulate_stats()
    base_ae.load_model(Path("models/"))

    latent_corrector = LatentCorrector(base_ae, db_path, corrector_config)
    latent_corrector.fit_model()
    latent_corrector.save(str(model_dir / "latent_corrector.h5"))
