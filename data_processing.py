import sys
import os
import pandas as pd
import pickle
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer

# Import our fix for library issues
import fix_textaugment

# Now import textaugment
from textaugment import EDA


def load_training_data(file_path):
    """
    Loads and processes the training data from a CSV file.

    Args:
        file_path (str): The path to the training CSV file.

    Returns:
        tuple: A tuple containing:
            - list: List of input queries (sentences).
            - list: List of multi-label tags.
    """
    df = pd.read_csv(file_path)
    df['tag'] = df['tag'].apply(lambda x: x.split(','))  # Convert tags to list
    sentences = df['query'].tolist()
    labels = df['tag'].tolist()

    # Debugging output
    print(f"[DEBUG] Loaded Sentences: {sentences}")
    print(f"[DEBUG] Loaded Labels: {labels}")

    return sentences, labels


def prepare_tokenizer(sentences, num_words=1000):
    """
    Prepares a tokenizer based on the provided sentences.

    Args:
        sentences (list): List of input queries.
        num_words (int): Maximum number of words to keep, based on word frequency.

    Returns:
        tuple: A tuple containing:
            - Tokenizer: A fitted Keras Tokenizer object.
            - np.ndarray: Padded sequences for the sentences.
    """
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, padding='post')
    return tokenizer, padded_sequences


def prepare_label_encoder(labels):
    """
    Prepares a MultiLabelBinarizer for the labels.

    Args:
        labels (list): List of multi-label tags.

    Returns:
        MultiLabelBinarizer: A fitted MultiLabelBinarizer object.
    """
    mlb = MultiLabelBinarizer()
    mlb.fit(labels)

    # Debugging output
    print(f"[DEBUG] Classes in MultiLabelBinarizer: {mlb.classes_}")

    return mlb


def save_pickle(obj, filename):
    """
    Saves a Python object to a file using pickle.

    Args:
        obj (object): The Python object to save.
        filename (str): The name of the output pickle file.
    """
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def main():
    """
    Main function to process training data and save tokenizer and label encoder.
    """
    data_path = 'data/training_data.csv'
    tokenizer_save_path = 'tokenizer.pkl'
    label_encoder_save_path = 'tag_encoder.pkl'

    sentences, labels = load_training_data(data_path)

    tokenizer, padded_sequences = prepare_tokenizer(sentences)
    mlb = prepare_label_encoder(labels)

    save_pickle(tokenizer, tokenizer_save_path)
    save_pickle(mlb, label_encoder_save_path)

    print(f"✅ Tokenizer and Label Encoder saved successfully!")


if __name__ == "__main__":
    main()