from scripts.train_imdb_llm import ReduceSum, SinActivation
import tensorflow as tf
from tensorflow import keras
import numpy as np

def load_and_summarize_model(model_path):
    # Load the model
    keras.config.enable_unsafe_deserialization()
    custom_objects = {"SinActivation": SinActivation, "ReduceSum": ReduceSum}
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    
    # Print basic model summary
    print("Basic Model Summary:")
    model.summary()
    
    print("\nDetailed Layer Information:")
    for i, layer in enumerate(model.layers):
        print(f"\nLayer {i}: {layer.name}")
        print(f"  Type: {type(layer).__name__}")
        print(f"  Input Shape: {layer.input_shape}")
        print(f"  Output Shape: {layer.output_shape}")
        print(f"  Trainable Parameters: {layer.count_params()}")
        
        # Print weights and biases statistics if available
        if layer.weights:
            for j, w in enumerate(layer.weights):
                weight_name = w.name
                weight_shape = w.shape
                weight_mean = np.mean(w.numpy())
                weight_std = np.std(w.numpy())
                print(f"  Weight {j}: {weight_name}")
                print(f"    Shape: {weight_shape}")
                print(f"    Mean: {weight_mean:.6f}")
                print(f"    Std Dev: {weight_std:.6f}")
    
    # Print total model statistics
    trainable_count = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    non_trainable_count = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    
    print(f"\nTotal trainable parameters: {trainable_count}")
    print(f"Total non-trainable parameters: {non_trainable_count}")
    print(f"Total parameters: {trainable_count + non_trainable_count}")

# Example usage
if __name__ == "__main__":
    model_path = "data/models/imdb_autoencoder.keras"
    load_and_summarize_model(model_path)