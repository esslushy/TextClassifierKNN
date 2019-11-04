from preprocessor import Preprocessor
from knn import KNearestNeighbor
from settings import CSV_LOCATION, LABEL_COLUMN, DATA_COLUMN

# Build processor
processor = Preprocessor(CSV_LOCATION, LABEL_COLUMN, DATA_COLUMN)
# Get data
data = processor.get_processed_data()
# Build classifier
knn = KNearestNeighbor(3, data, processor.labels)
# Make predictions
print(knn.predict_classes(data[0]['data']))