"""
Model Building Module

This module provides a function to build a simple neural network model for multi-label text classification
using Keras' Sequential API.
"""

from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Embedding, GlobalAveragePooling1D, Dense


def build_model(vocab_size, output_dim):
    """
    Builds and compiles a Keras Sequential model for multi-label classification tasks.

    The architecture consists of:
    - An Embedding layer to learn word representations.
    - A GlobalAveragePooling1D layer to reduce variable-length sequences into fixed-length vectors.
    - A hidden Dense layer with ReLU activation.
    - An output Dense layer with sigmoid activation for multi-label prediction.

    Args:
        vocab_size (int): Size of the vocabulary (number of unique tokens).
        output_dim (int): Number of target labels/classes (i.e., the dimensionality of the output layer).

    Returns:
        keras.Model: A compiled Keras model ready for training.
    """
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=64))  # Embedding layer
    model.add(GlobalAveragePooling1D())  # Reduces sequence to fixed-size vector
    model.add(Dense(64, activation='relu'))  # Hidden fully connected layer
    model.add(Dense(output_dim, activation='sigmoid'))  # Output layer for multi-label classification

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
