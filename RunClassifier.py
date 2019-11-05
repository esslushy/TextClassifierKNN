from preprocessor import Preprocessor
from knn import KNearestNeighbor
from settings import CSV_LOCATION, LABEL_COLUMN, DATA_COLUMN
from sklearn.model_selection import KFold
import numpy as np

# Build processor
processor = Preprocessor(CSV_LOCATION, LABEL_COLUMN, DATA_COLUMN)

# Get data
data = processor.get_processed_data()
data = np.array(data)
print('Processed Data')

# KFold split and validation
kf = KFold(n_splits=10)

# Accuracy storing array
accuracies = []

# Run over each fold
for train_index, test_index in kf.split(data):
    # Get train and test data
    print("TRAIN:", train_index, "TEST:", test_index)
    train, test = data[train_index], data[test_index]
    # Build classifier on data
    knn = KNearestNeighbor(5, train, processor.labels)
    # Make predictions
    predictions = [knn.predict_class(point['data']) for point in test]
    # Compare to actual labels
    correct_labels = [int(prediction == label['label']) for prediction, label in zip(predictions, test)]
    # Calculate accuracy
    accuracy = sum(correct_labels)/len(correct_labels)
    print(accuracy)
    accuracies.append(accuracy)

# Print out how well it performed
print(accuracies)