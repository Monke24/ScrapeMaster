"""
Tag Prediction Module

This script defines a base `Predictor` class and a `TagPredictor` subclass for making multi-label
predictions using a trained Keras model, tokenizer, and encoder.
"""

import pickle
import numpy as np
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras.src.models.sequential import Sequential


class Predictor:
    """
    Abstract base class for prediction models.
    Demonstrates **polymorphism** by defining a common interface (`predict` and `load_resources`)
    that subclasses must implement.
    """

    def __init__(self):
        # === Encapsulation ===
        # Use of underscore-prefixed variables (_model, _tokenizer, _encoder)
        # indicates they are intended to be treated as private to the class.
        self._model = None
        self._tokenizer = None
        self._encoder = None

    def load_resources(self):
        # === Polymorphism ===
        # This method is meant to be overridden in a subclass.
        raise NotImplementedError("Must be implemented in subclass.")

    def predict(self, query):
        # === Polymorphism ===
        # Again, this is designed to be overridden with specific logic in a subclass.
        raise NotImplementedError("Must be implemented in subclass.")


class TagPredictor(Predictor):  # === Inheritance ===
    """
    Concrete implementation of Predictor for multi-label HTML tag prediction.
    """

    def __init__(self, model_path, tokenizer_path, encoder_path, max_len=100, threshold=0.2):
        super().__init__()  # Call parent constructor
        # === Encapsulation ===
        # Internal variables are kept private using underscore naming.
        self._model_path = model_path
        self._tokenizer_path = tokenizer_path
        self._encoder_path = encoder_path
        self._max_len = max_len
        self._threshold = threshold
        self.load_resources()

    def load_resources(self):
        # === Polymorphism ===
        # Overriding the abstract method from the base class with specific implementation.
        with open(self._tokenizer_path, "rb") as f:
            self._tokenizer = pickle.load(f)
        with open(self._encoder_path, "rb") as f:
            self._encoder = pickle.load(f)
        self._model = load_model(self._model_path)

    def predict(self, query: str):
        # === Polymorphism ===
        # This method overrides the base class method with specific behavior for tag prediction.
        sequence = self._tokenizer.texts_to_sequences([query])
        padded = pad_sequences(sequence, maxlen=self._max_len, padding='post')

        if not isinstance(self._model, Sequential):
            return
        prediction = self._model.predict(padded)
        prediction = np.array(prediction)

        binary_prediction = (prediction > self._threshold).astype(int)

        # Debug prints
        print(f"[DEBUG] Raw prediction: {prediction}")
        print(f"[DEBUG] Binary prediction: {binary_prediction}")
        print(f"[DEBUG] Type of binary_prediction: {type(binary_prediction)}")
        print("test")
        print(f"[DEBUG] Shape of binary_prediction: {binary_prediction.shape}")
        print("test2")

        if np.sum(binary_prediction) == 0:
            return []

        tags = self._encoder.inverse_transform(binary_prediction)
        return tags[0] if tags and len(tags) > 0 else []


# === Example Usage ===
if __name__ == "__main__":
    predictor = TagPredictor(
        model_path="model.h5",
        tokenizer_path="tokenizer.pkl",
        encoder_path="tag_encoder.pkl"
    )

    while True:
        user_input = input("\nEnter your query (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        try:
            tags = predictor.predict(user_input)
            print("✅ Predicted Tags:", tags)
        except Exception as e:
            print("❌ An error occurred:", e)
