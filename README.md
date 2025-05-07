# ScrapeMaster
Read Me Description
My package is a web scraping tool that software developers can use to quickly access information from websites to lower the time it takes them to research. I made this because software developers can struggle to find information and research promptly, but my package gives them a quick and efficient way to get the information they want from websites. This will allow them to have more time making and debugging their project so that they aren’t tight on deadlines and can even get ahead of them. 

Libraries
It uses BS4, Requests, Pickle, Tensorflow, Keras, Sklearn, Pandas, and PYQT6. I used BS4 and Requests to manage the webscraping of the URL that the user provides. Then I used Pickle to serialize and deserialize the Multi-Label Binarizer and the Tokenizer. I then used Tensorflow, Keras, and Sklearn to build and train the AI model that takes in an input from the user and tells BS4 what to look for in the website for more efficiency. Finally, I used PyQT6 to make the GUI so the Software Developers can use the project on their PC’s. 

Code made by me vs not made by me
Even though I used so many different libraries, I still coded almost all of the package. In the Data Processing File, I coded all the functions, which include load_training_data, encoder, and prepare_tokenizer. In my model_utils, I coded the entire file, including the build_model function. In the train_data file, I coded all of it, including the load_data, prepare_data, and train_model functions. In the prediction.py file, I coded all of it, including the Predictor and Tag Predictors classes, as well as the load_resources and predict functions. In the final file, mainapp.py, I coded almost the entire file except for one part that I used for the webscraping using Bs4. 

Docstrings

Data_processing.py

Loads and processes the training data from a CSV file.

Args:
    file_path (str): The path to the training CSV file.

Returns:
    tuple: A tuple containing:
        - list: List of input queries (sentences).
        - list: List of multi-label tags.
Prepares a tokenizer based on the provided sentences.

Args:
    sentences (list): List of input queries.
    num_words (int): Maximum number of words to keep, based on word frequency.

Returns:
    tuple: A tuple containing:
        - Tokenizer: A fitted Keras Tokenizer object.
        - np.ndarray: Padded sequences for the sentences.
Prepares a MultiLabelBinarizer for the labels.

Args:
    labels (list): List of multi-label tags.

Returns:
    MultiLabelBinarizer: A fitted MultiLabelBinarizer object.
Saves a Python object to a file using pickle.

Args:
    obj (object): The Python object to save.
    filename (str): The name of the output pickle file.
Main function to process training data and save tokenizer and label encoder.

—--------------------------------------------------------------------------------------------------

Model_utils.py

Model Building Module

This module provides a function to build a simple neural network model for multi-label text classification
using Keras' Sequential API.
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

—--------------------------------------------------------------------------------------------------

Train_data.py

Trains the AI model using the data that has already been processed in the data_processing file.

The script has no classes but contains many different functions that work together to train the data.
These include load_data and train_model.

Example Usage: When you run the code, it will process the data and start training the model on it with early stopping to prevent the model from memorizing the data.
Loads the training data from a CSV file.

Args:
    csv_path (str): The file path to the CSV file containing the training data.

Returns:
    pd.DataFrame: The loaded training data as a Pandas DataFrame.
Prepares the input data by tokenizing and padding the queries, and encoding the tags.

Args:
    data (pd.DataFrame): The training data.
    tokenizer (Tokenizer): A Keras Tokenizer object used to tokenize the queries.
    mlb (MultiLabelBinarizer): A fitted MultiLabelBinarizer object to encode tags.

Returns:
    tuple: A tuple containing:
        - np.ndarray: Padded tokenized input queries.
        - np.ndarray: Encoded tags.
Trains a machine learning model for tag prediction, saves the model, and handles the training callbacks.

Args:
    csv_path (str): The file path to the training data CSV file.
    tokenizer_path (str): The file path to the pre-trained tokenizer.
    tag_encoder_path (str): The file path to the pre-trained label encoder.
    model_save_path (str): The file path where the trained model will be saved.

Raises:
    ValueError: If the tag matrix doesn't have the expected 2D shape.
Learning rate scheduler function that decreases the learning rate by 10% every 10 epochs.

Args:
    epoch (int): The current epoch number.
    lr (float): The current learning rate.

Returns:
    float: The adjusted learning rate.

—--------------------------------------------------------------------------------------------------

Predict.py

Tag Prediction Module

This script defines a base `Predictor` class and a `TagPredictor` subclass for making multi-label
predictions using a trained Keras model, tokenizer, and encoder.

Example usage:
    Run this file directly to input natural language queries and receive predicted HTML tag labels.
Abstract base class for prediction models.
Concrete implementation of Predictor for multi-label HTML tag prediction.
Load required resources such as model, tokenizer, and encoder.
Must be implemented in subclass.
Make a prediction based on a query.
Must be implemented in subclass.
Initialize and load model, tokenizer, and encoder.

Args:
    model_path (str): Path to the trained Keras model (.h5 file).
    tokenizer_path (str): Path to the tokenizer pickle file.
    encoder_path (str): Path to the label encoder pickle file.
    max_len (int): Maximum sequence length for padding.
    threshold (float): Prediction confidence threshold.
Load tokenizer, encoder, and model from their respective files.
Predict tags for a given natural language query.

Args:
    query (str): User input query.

Returns:
    list: Predicted tags or an empty list if none surpass the threshold.

—--------------------------------------------------------------------------------------------------

Main_app.py

ScrapeMaster (v. Lebron)

This application uses a trained ML model to predict HTML tags based on a user query,
then scrapes the content of a given URL for those tags and displays the results.

Technologies used:
- TensorFlow/Keras for tag prediction
- BeautifulSoup for web scraping
- PyQt6 for GUI interface
Clean the user input using regex to remove punctuation and convert to lowercase.

Args:
    query (str): Raw user query.

Returns:
    str: Cleaned query string.
A PyQt6-based GUI application for scraping HTML elements predicted from user queries.
Set up the user interface.
Handle the scrape button click: fetch page, predict tags, and extract content.

