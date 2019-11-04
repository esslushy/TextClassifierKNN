import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import namedtuple
import numpy as np

# Set up named tuple for easy access of data

class Preprocessor():
    """
      Creates a class that consumes a csv file and produces an dictionary
      that can be consumed by the knn classifier to train and make
      predictions

      Args:
        file_location: location of the csv file  
        label_column: the string that identifies the label column in the csv  
        data_column: the string that identifies the data column in the csv  
    """
    def __init__(self, file_location, label_column, data_column):
        self.dataframe = pd.read_csv(file_location)[[label_column, data_column]]
        self.label = label_column
        self.data = data_column
        self.vectorizer = TfidfVectorizer()
        self.remove_nan()
        self.fit_vectorizer()
        self.labels = np.unique(self.dataframe[self.label])

    def remove_nan(self):
        self.dataframe = self.dataframe.dropna()

    def fit_vectorizer(self):
        # Prepares the vectorizer on the data 
        corpus = self.dataframe[self.data].values
        self.vectorizer.fit(corpus)
    
    def get_processed_data(self):
        # Process the data and returns it as a dictionary with the label and processed data in pairs.
        processed_data = self.vectorizer.transform(self.dataframe[self.data].values)
        labels = self.dataframe[self.label].values
        return [{ 'label' : labels[i], 'data' : processed_data[i] } for i in range(len(labels))]