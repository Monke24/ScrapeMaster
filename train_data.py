'''
Trains the AI model using the data that has already been processed in the data_processing file.

The script has no classes but contains many different functions that work together to train the data.
These include load_data and train_model.

Example Usage: When you run the code, it will process the data and start training the model on it with early stopping to prevent the model from memorizing the data.
'''


import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.callbacks import EarlyStopping, LearningRateScheduler
from keras._tf_keras.keras.optimizers import Adam
import os
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from model_utils import build_model


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Loads the training data from a CSV file.

    Args:
        csv_path (str): The file path to the CSV file containing the training data.

    Returns:
        pd.DataFrame: The loaded training data as a Pandas DataFrame.
    """
    data: pd.DataFrame = pd.read_csv(csv_path)
    print(type(data))
    return data


def prepare_data(data, tokenizer, mlb):
    """
    Prepares the input data by tokenizing and padding the queries, and encoding the tags.

    Args:
        data (pd.DataFrame): The training data.
        tokenizer (Tokenizer): A Keras Tokenizer object used to tokenize the queries.
        mlb (MultiLabelBinarizer): A fitted MultiLabelBinarizer object to encode tags.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Padded tokenized input queries.
            - np.ndarray: Encoded tags.
    """
    sentences = data["query"].values
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, maxlen=100, padding='post')

    tags = data["tag"].apply(lambda x: x.split(","))
    tag_matrix = mlb.transform(tags)
    
    return padded_sequences, tag_matrix


def train_model(csv_path, tokenizer_path, tag_encoder_path, model_save_path):
    """
    Trains a machine learning model for tag prediction, saves the model, and handles the training callbacks.

    Args:
        csv_path (str): The file path to the training data CSV file.
        tokenizer_path (str): The file path to the pre-trained tokenizer.
        tag_encoder_path (str): The file path to the pre-trained label encoder.
        model_save_path (str): The file path where the trained model will be saved.

    Raises:
        ValueError: If the tag matrix doesn't have the expected 2D shape.
    """
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    with open(tag_encoder_path, "rb") as f:
        mlb = pickle.load(f)

    # Load and prepare data
    data = load_data(csv_path)
    padded_sequences, tag_matrix = prepare_data(data, tokenizer, mlb)
    if not isinstance(tag_matrix, np.ndarray):
        print("err: not NDArray, line 85, train_data")

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        padded_sequences, tag_matrix, test_size=0.2, random_state=42
    )

    # Build the model
    vocab_size = len(tokenizer.word_index) + 1

    print("test")
    if len(tag_matrix.shape) != 2:
        raise ValueError(f"Expected tag_matrix to be 2D, but got shape {tag_matrix.shape}")
    output_dim = tag_matrix.shape[1]


    model = build_model(vocab_size, output_dim)

    # Optimizer with learning rate decay
    learning_rate = 0.001
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy']) # type: ignore

    # Learning rate scheduler callback
    def lr_scheduler(epoch, lr):
        """
        Learning rate scheduler function that decreases the learning rate by 10% every 10 epochs.

        Args:
            epoch (int): The current epoch number.
            lr (float): The current learning rate.

        Returns:
            float: The adjusted learning rate.
        """
        if epoch % 10 == 0 and epoch > 0:
            lr = lr * 0.1  # Decrease learning rate by 10% every 10 epochs
        return lr

    lr_callback = LearningRateScheduler(lr_scheduler)

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model with callbacks
    model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), batch_size=16, 
              callbacks=[early_stopping, lr_callback])

    # Save the model
    model.save(model_save_path)

    print(f"✅ Model trained and saved as {model_save_path}!")


# Running the training process
if __name__ == "__main__":
    base_path = os.path.dirname(__file__)
    csv_path = os.path.join(base_path, 'data', 'training_data.csv')
    tokenizer_path = "tokenizer.pkl"
    tag_encoder_path = "tag_encoder.pkl"
    model_save_path = "model.h5"
    
    train_model(csv_path, tokenizer_path, tag_encoder_path, model_save_path)
