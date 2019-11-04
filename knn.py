from sklearn.metrics.pairwise import cosine_similarity

class KNearestNeighbor():
    """
      Creates a class that consumes a csv file and produces an dictionary
      that can be consumed by the knn classifier to train and make
      predictions

      Args:
        k: Number of neighbors to calculate relation to.
        data: An array of data in a label, data dictionary or named tuple format
        labels: The labels that exist within the dataset
    """
    def __init__(self, k, data, labels):
        self.k = k
        self.data = data
        self.labels = labels

    def _insertion_sort(self, arr):
        """
          Sorts an array from greatest to least in a compatible way with how data is passed in this class

          Args:
            arr: Array to be sorted in format [{similarity, label}]
        """
        for i in range(1, len(arr)):
            key = arr[i]
            position = i-1
            while position > 0 and key['similarity'] > arr[position]['similarity']:
                arr[position+1] = arr[position]
                position -= 1
            arr[position+1] = key
        return arr

    def _get_distance_to_data(self, point):
        """
          Computes the distances of all data points to the passed data point

          Args:
            point: The point to calculate similarity to.
        """
        similarities = []
        for data in self.data:
            # Calculate similarity 
            similarity = cosine_similarity(point, data['data'])[0][0]
            # Add to array
            similarities.append({ 'similarity' : similarity, 'label' : data['label'] })
        return similarities
    
    def _find_nearest_neighbors(self, new_point):
        """
          Gets k number of nearest neighbors to the data point

          Args:
           new_point: The new point to find the neighbors to
        """
        # Get similarities
        similarities = self._get_distance_to_data(new_point)
        # Sort from greatest to least
        similarities = self._insertion_sort(similarities)
        # Return the first k numbers as they will be the most similar
        return similarities[:self.k]
    
    def predict_classes(self, new_point):
        """
          Returns the predicted probabilities for each label

          Args:
            new_point: The new point to make a prediction on
        """
        # Get neighbors
        neighbors = self._find_nearest_neighbors(new_point)
        # Assemble dictionary of classes
        classes = {}
        for label in self.labels:
            classes[label] = 0
        # Count each neighbor in each label
        for neighbor in neighbors:
            classes[neighbor['label']] = classes[neighbor['label']] + 1
        # Divide by k for probabilities
        for label in self.labels:
            classes[label] = classes[label] / self.k
        return classes