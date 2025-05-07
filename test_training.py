import unittest
import os
import pickle
from train_data import load_data, prepare_data, train_model

class TestTrainingData(unittest.TestCase):
    """
    Unit tests for training data preparation and model training.
    """

    def setUp(self):
        """
        Set up file paths and ensure required files exist.
        """
        self.base_path = os.path.dirname(__file__)
        self.data_path = os.path.join(self.base_path, 'data', 'training_data.csv')
        self.tokenizer_path = os.path.join(self.base_path, 'tokenizer.pkl')
        self.tag_encoder_path = os.path.join(self.base_path, 'tag_encoder.pkl')
        self.model_save_path = os.path.join(self.base_path, 'model.h5')

        # Ensure that the CSV exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Training data file not found at {self.data_path}")

        # Ensure tokenizer and tag encoder exist
        if not os.path.exists(self.tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found at {self.tokenizer_path}")

        if not os.path.exists(self.tag_encoder_path):
            raise FileNotFoundError(f"Tag encoder file not found at {self.tag_encoder_path}")

    def test_data_loading(self):
        """
        Test that data loads correctly and is not empty.
        """
        data = load_data(self.data_path)
        self.assertFalse(data.empty, "Loaded data should not be empty.")
        self.assertIn("query", data.columns, "'query' column should exist in the data.")
        self.assertIn("tag", data.columns, "'tag' column should exist in the data.")

    def test_data_preparation(self):
        """
        Test that data preparation (tokenization and label encoding) works correctly.
        """
        with open(self.tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)

        with open(self.tag_encoder_path, "rb") as f:
            mlb = pickle.load(f)

        data = load_data(self.data_path)
        padded_sequences, tag_matrix = prepare_data(data, tokenizer, mlb)

        self.assertEqual(len(padded_sequences), len(tag_matrix), 
                         "The number of padded sequences should match the number of labels.")
        self.assertGreater(padded_sequences.shape[1], 0, 
                           "Padded sequences should have a positive sequence length.")

    def test_model_training(self):
        """
        Test that the model training function runs and saves a model file.
        """
        # Train the model
        train_model(self.data_path, self.tokenizer_path, self.tag_encoder_path, self.model_save_path)

        # Check if model was saved
        self.assertTrue(os.path.exists(self.model_save_path), "Trained model file should exist.")

if __name__ == '__main__':
    unittest.main()
