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
    def __init__(self, base_autoencoder: RowAutoencoder, latent_dim: int, correction_network: Model):
        super().__init__()
        self.base_autoencoder = base_autoencoder
        self.latent_dim = latent_dim
        self.correction_network = correction_network
        self.autoencoder = self._build_autoencoder_with_correction()

    def _build_autoencoder_with_correction(self):
        fields = self.base_autoencoder.fields
        encoder_inputs = {}
        encoder_outputs = {}
        decoders = {}
        decoder_outputs = {}

        for field in fields:
            inp = tf.keras.Input(shape=field.input_shape, name=f"{field.name}_input", dtype=field.input_dtype)
            encoder_inputs[field.name] = inp

        for field in fields:
            encoder = field.build_encoder(self.latent_dim)
            encoder_outputs[field.name] = encoder(encoder_inputs[field.name])
            decoders[field.name] = field.build_decoder(self.latent_dim)

        combined = layers.Concatenate()(list(encoder_outputs.values()))
        x = layers.Dense(self.latent_dim * 8, activation='gelu', name="latent_dense1")(combined)
        x = layers.Dense(self.latent_dim * 2, activation='gelu', name="latent_dense2")(x)
        latent = layers.Dense(self.latent_dim, activation='linear', name="latent_dense3")(x)
        latent = layers.LayerNormalization(name="latent_ln")(latent)

        corrected = self.correction_network(latent)

        for field in fields:
            decoder_outputs[field.name] = decoders[field.name](corrected)

        inputs_list = [encoder_inputs[field.name] for field in fields]
        outputs_list = [decoder_outputs[field.name] for field in fields]
        return Model(inputs=inputs_list, outputs=outputs_list, name="RecursiveCorrectionAutoencoder")

    def call(self, inputs, training=False):
        return self.autoencoder(inputs, training=training)

if __name__ == "__main__":
    config = project_config["autoencoder"]
    data_dir = Path(project_config["data_dir"])
    db_path = data_dir / 'imdb.db'

    base_ae = TitlesAutoencoder(config, db_path)
    base_ae.accumulate_stats()
    latent_dim = config["latent_dim"]
    correction_net = build_correction_network(latent_dim, hidden_units=256, num_layers=3)

    rc_autoencoder = RecursiveCorrectionAutoencoder(base_ae, latent_dim, correction_net)
    rc_autoencoder.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
    loss_dict = base_ae.get_loss_dict()
    loss_weights = base_ae.get_loss_weights_dict()
    rc_autoencoder.compile(optimizer=optimizer, loss=loss_dict, loss_weights=loss_weights)

    dataset = base_ae._build_dataset(db_path)
    rc_autoencoder.fit(dataset, epochs=5)

    rc_autoencoder.save("recursive_correction_autoencoder.h5")
