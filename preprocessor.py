import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class Preprocessor():
    def __init__(self, file_location, label_column, data_column):
        self.dataframe = pd.read_csv(file_location)[[label_column, data_column]]
        self.label = label_column
        self.data = data_column
        self.vectorizer = TfidfVectorizer()
        self.fit_vectorizer()

    def fit_vectorizer(self):
        corpus = self.dataframe[self.data].values
        self.vectorizer.fit(corpus)
    
    def get_processed_data(self):
        processed_data = self.vectorizer.transform(self.dataframe[self.data].values)
        labels = self.dataframe[self.label].values
        return {labels[i]: processed_data[i] for i in range(len(labels))}