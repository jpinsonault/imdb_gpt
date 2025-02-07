import tensorflow as tf
from tensorflow.keras import layers, Model
from pathlib import Path
from config import project_config
from autoencoder.schema import RowAutoencoder
from autoencoder.imdb_row_autoencoders import TitlesAutoencoder

def build_correction_network(latent_dim, hidden_units=256, num_layers=3):
    inp = tf.keras.Input(shape=(latent_dim,), name="latent_correction_input")
    x = inp
    for i in range(num_layers):
        x = layers.Dense(hidden_units, activation=tf.sin, name=f"siren_dense_{i}")(x)

    correction = layers.Dense(latent_dim, activation='linear', name="corrected_latent")(x)

    corrected_input = layers.Add()([inp, correction])
    return Model(inp, corrected_input, name="correction_network")

class RecursiveCorrectionAutoencoder(tf.keras.Model):
    def __init__(self, base_autoencoder: RowAutoencoder, latent_dim: int):
        super().__init__()
        self.base_autoencoder = base_autoencoder
        self.latent_dim = latent_dim
        # TODO
        
if __name__ == "__main__":
    config = project_config["autoencoder"]
    latent_dim = config["latent_dim"]
    data_dir = Path(project_config["data_dir"])
    db_path = data_dir / 'imdb.db'

    base_ae = TitlesAutoencoder(config, db_path)
    base_ae.accumulate_stats()
    base_ae.load_model(Path("models/"))

    rc_autoencoder = CorrectionAutoencoder(base_ae, latent_dim)
    rc_autoencoder.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
    loss_dict = base_ae.get_loss_dict()
    loss_weights = base_ae.get_loss_weights_dict()
    rc_autoencoder.compile(optimizer=optimizer, loss=loss_dict, loss_weights=loss_weights)

    dataset = base_ae._build_dataset(db_path)
    rc_autoencoder.fit(dataset, epochs=5)

    rc_autoencoder.save("recursive_correction_autoencoder.h5")
