import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class LogisticMapDenseLayer(tf.keras.layers.Layer):
    def __init__(self, units, cache_size=1000, **kwargs):
        super(LogisticMapDenseLayer, self).__init__(**kwargs)
        self.units = units
        self.cache_size = cache_size
    
    def build(self, input_shape):
        self.r = self.add_weight(
            name='r', shape=(), 
            initializer=tf.random_uniform_initializer(minval=0.0, maxval=1.0),
            constraint=tf.keras.constraints.MinMaxNorm(min_value=0.0, max_value=1.0),
            trainable=True
        )
        self.num_weights = input_shape[-1] * self.units
        self.bias = self.add_weight(
            name='bias', shape=(self.units,), initializer='zeros', trainable=True
        )
        self.cache = self.add_weight(
            name='cache', shape=(self.cache_size, self.num_weights),
            initializer='zeros', trainable=False
        )
        self.cache_index = self.add_weight(
            name='cache_index', shape=(), initializer='zeros',
            dtype=tf.int32, trainable=False
        )
        super(LogisticMapDenseLayer, self).build(input_shape)
    
    def call(self, inputs):
        r_mapped = 3.57 + (4.0 - 3.57) * self.r
        cache_r = tf.gather(self.cache, self.cache_index)[0]
        
        def update_cache():
            x = tf.random.uniform(shape=(self.num_weights,), minval=0, maxval=1)
            
            def logistic_map_step(x, _):
                return r_mapped * x * (1 - x)
            
            weights = tf.scan(logistic_map_step, tf.range(self.num_weights), initializer=x)
            weights = tf.reshape(weights, (-1,))  # Flatten the result
            
            new_cache_index = (self.cache_index + 1) % self.cache_size
            self.cache.assign(tf.tensor_scatter_nd_update(
                self.cache, [[new_cache_index]], [weights]
            ))
            self.cache_index.assign(new_cache_index)
            return weights
        
        weights = tf.cond(
            tf.math.not_equal(cache_r, r_mapped),
            update_cache,
            lambda: tf.gather(self.cache, self.cache_index)
        )
        
        weights = tf.reshape(weights, (inputs.shape[-1], self.units))
        return tf.matmul(inputs, weights) + self.bias

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

def generate_data(n_samples, n_features, n_classes):
    X = []
    y = []
    for i in range(n_classes):
        mean = np.random.randn(n_features)
        cov = np.random.randn(n_features, n_features)
        cov = np.dot(cov, cov.T)  # Ensure covariance matrix is positive semi-definite
        X.append(np.random.multivariate_normal(mean, cov, n_samples // n_classes))
        y.append(np.full(n_samples // n_classes, i))
    return np.vstack(X), np.hstack(y)

# Generate data
n_samples = 10000
n_features = 10
n_classes = 5
X, y = generate_data(n_samples, n_features, n_classes)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
inputs = tf.keras.Input(shape=(n_features,))
x = LogisticMapDenseLayer(64, cache_size=100)(inputs)
x = tf.keras.layers.Activation('relu')(x)
x = LogisticMapDenseLayer(32, cache_size=100)(x)
x = tf.keras.layers.Activation('relu')(x)
outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Print final r values
for i, layer in enumerate(model.layers):
    if isinstance(layer, LogisticMapDenseLayer):
        print(f"LogisticMapDenseLayer {i} final r value: {layer.r.numpy():.4f}")    