import random
import re
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention, LayerNormalization, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.losses import categorical_crossentropy


leaky_relu = layers.LeakyReLU(alpha=0.2)


def multiheaded_generative_matrix_attention(num_filters, embedding_dim, num_heads, name):
    """a multi-head attention layer."""
    
    assert num_filters % num_heads == 0, "num_filters must be divisible by num_heads"
    
    depth = num_filters // num_heads

    value_conv = layers.Conv1D(filters=num_filters, kernel_size=1, padding="same", activation=None, use_bias=False, name=f"{name}_value_conv")
    key_conv = layers.Conv1D(filters=num_filters, kernel_size=1, padding="same", activation=None, use_bias=False, name=f"{name}_key_conv")
    query_conv = layers.Conv1D(filters=num_filters, kernel_size=1, padding="same", activation=None, use_bias=False, name=f"{name}_query_conv")
    
    regularize_attention = layers.Lambda(lambda x: x / tf.math.sqrt(tf.cast(depth, tf.float16)))
    
    projection_conv1 = layers.Conv1D(filters=embedding_dim*4, kernel_size=1, padding="same", activation=None, name=f"{name}_projection_conv")
    projection_conv2 = layers.Conv1D(filters=embedding_dim, kernel_size=1, padding="same", activation=leaky_relu, name=f"{name}_projection_conv2")
    layer_norm = layers.LayerNormalization(name=f"{name}_layer_norm")

    def split_heads(x, batch_size, input_length):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, input_length, num_heads, depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def func(input_tensor):
        batch_size = tf.shape(input_tensor)[0]
        input_length = tf.shape(input_tensor)[1]
        
        # Compute the value, key, and query tensors
        value_output = value_conv(input_tensor)
        key_output = key_conv(input_tensor)
        query_output = query_conv(input_tensor)
        
        query_output = layers.LayerNormalization()(query_output)
        key_output = layers.LayerNormalization()(key_output)
        
        # Split heads
        value_output = split_heads(value_output, batch_size, input_length)
        key_output = split_heads(key_output, batch_size, input_length)
        query_output = split_heads(query_output, batch_size, input_length)
        
        # Compute the attention mask
        attention_mask = tf.matmul(query_output, key_output, transpose_b=True)
        attention_mask = regularize_attention(attention_mask)
        attention_mask = layers.Softmax(axis=-1, name=f"attention_mask_{name}")(attention_mask)
        
        # Apply the attention mask to the value tensor
        attended_values = tf.matmul(attention_mask, value_output)
        
        # Merge heads
        attended_values = tf.transpose(attended_values, perm=[0, 2, 1, 3])
        attended_values = tf.reshape(attended_values, (batch_size, -1, num_filters))
        
        attended_values = layers.Add()([attended_values, input_tensor])
        
        # Apply a projection to the attended values
        projected_values = projection_conv1(attended_values)
        projected_values = projection_conv2(projected_values)
        projected_values = leaky_relu(layers.LayerNormalization()(projected_values))

        # Add the projected values to the input tensor
        output = layers.Add()([attended_values, projected_values])
        output = layers.Add()([output, input_tensor])
        output = layer_norm(output)
        
        return output

    return func



def generative_matrix_attention(num_filters, embedding_dim, name):
    """a classic attention is all you need matrix attention layer."""
    value_conv = layers.Conv1D(filters=embedding_dim, kernel_size=1, padding="same", activation=None, use_bias=False, name=f"{name}_value_conv")
    key_conv = layers.Conv1D(filters=num_filters, kernel_size=1, padding="same", activation=None, use_bias=False, name=f"{name}_key_conv")
    query_conv = layers.Conv1D(filters=num_filters, kernel_size=1, padding="same", activation=None, use_bias=False, name=f"{name}_query_conv")
    
    regularize_attention = layers.Lambda(lambda x: x / tf.math.sqrt(tf.cast(x.shape[-1], tf.float16)))
    
    projection_conv1 = layers.Conv1D(filters=embedding_dim*4, kernel_size=1, padding="same", activation=None, name=f"{name}_projection_conv")
    projection_conv2 = layers.Conv1D(filters=embedding_dim, kernel_size=1, padding="same", activation=leaky_relu, name=f"{name}_projection_conv2")
    layer_norm = layers.LayerNormalization(name=f"{name}_layer_norm")

    def func(input_tensor):
        # Compute the value, key, and query tensors
        value_output = value_conv(input_tensor)
        key_output = key_conv(input_tensor)
        query_output = query_conv(input_tensor)
        
        query_output = layers.LayerNormalization()(query_output)
        key_output = layers.LayerNormalization()(key_output)
        
        # Compute the attention mask
        attention_mask = tf.matmul(query_output, key_output, transpose_b=True)
        attention_mask = regularize_attention(attention_mask)
        attention_mask = layers.Softmax(axis=-1, name=f"attention_mask_{name}")(attention_mask)
        
        # Apply the attention mask to the value tensor
        attended_values = tf.matmul(attention_mask, value_output)
        
        attended_values = layers.Add()([attended_values, input_tensor])
        
        # Apply a projection to the attended values
        projected_values = projection_conv1(attended_values)
        projected_values = projection_conv2(projected_values)
        projected_values = layers.LayerNormalization()(projected_values)
        projected_values = leaky_relu(projected_values)

        # Add the projected values to the input tensor
        output = layers.Add()([attended_values, projected_values])
        output = layers.Add()([output, input_tensor])
        output = layer_norm(output)
        
        return output

    return func