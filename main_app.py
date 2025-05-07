"""
ScrapeMaster (v. Lebron)

This application uses a trained ML model to predict HTML tags based on a user query,
then scrapes the content of a given URL for those tags and displays the results.

Technologies used:
- TensorFlow/Keras for tag prediction
- BeautifulSoup for web scraping
- PyQt6 for GUI interface
"""

import sys
import requests
import re
from bs4 import BeautifulSoup
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit,
    QLineEdit, QLabel, QPlainTextEdit
)

from predict import TagPredictor  # <-- Import your prediction class


# === Initialize Tag Predictor ===
predictor = TagPredictor(
    model_path="model.h5",
    tokenizer_path="tokenizer.pkl",
    encoder_path="tag_encoder.pkl",
    max_len=100,
    threshold=0.5
)


def clean_input(query):
    """
    Clean the user input using regex to remove punctuation and convert to lowercase.

    Args:
        query (str): Raw user query.

    Returns:
        str: Cleaned query string.
    """
    cleaned_query = re.sub(r'[^\w\s]', '', query)
    return cleaned_query.lower()


class WebScraperApp(QWidget):
    """
    A PyQt6-based GUI application for scraping HTML elements predicted from user queries.
    """

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        """
        Set up the user interface.
        """
        self.setWindowTitle("ScrapeMaster (v. Lebron)")
        self.setGeometry(100, 100, 600, 400)

        layout = QVBoxLayout()

        self.url_label = QLabel("Enter URL:")
        layout.addWidget(self.url_label)

        self.url_input = QLineEdit(self)
        layout.addWidget(self.url_input)

        self.query_label = QLabel("What are you looking for?")
        layout.addWidget(self.query_label)

        self.query_input = QPlainTextEdit(self)
        layout.addWidget(self.query_input)

        self.scrape_button = QPushButton("Scrape Website", self)
        self.scrape_button.clicked.connect(self.scrape_website)
        layout.addWidget(self.scrape_button)

        self.result_text = QTextEdit(self)
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)

        self.setLayout(layout)

    def scrape_website(self):
        """
        Handle the scrape button click: fetch page, predict tags, and extract content.
        """
        url = self.url_input.text().strip()
        user_query = self.query_input.toPlainText().strip()

        if not url or not user_query:
            self.result_text.setText("Please enter both a URL and a query.")
            return

        try:
            cleaned_query = clean_input(user_query)
            predicted_tags = predictor.predict(cleaned_query)

            if not predicted_tags:
                self.result_text.setText("No tags were predicted for this query.")
                return

            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            results = []
            for tag in predicted_tags:
                elements = soup.find_all(tag)
                for elem in elements:
                    text = elem.get_text(strip=True)
                    if text:
                        results.append(f"{tag.upper()}: {text}")

            if results:
                self.result_text.setText("\n".join(results))
            else:
                self.result_text.setText("No matching elements found.")

        except requests.exceptions.RequestException as e:
            self.result_text.setText(f"Error fetching the website: {e}")
        except Exception as e:
            self.result_text.setText(f"An error occurred: {e}")


# === Run the Application ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WebScraperApp()
    window.show()
    sys.exit(app.exec())
