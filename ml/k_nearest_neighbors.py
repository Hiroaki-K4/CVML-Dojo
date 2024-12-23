from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Store the training data."""
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        """Predict the class for each data point in X."""
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        """Predict the class of a single data point."""
        # Compute distances to all training points
        distances = np.linalg.norm(self.X_train - x, axis=1)
        # Get indices of the k nearest neighbors
        k_indices = np.argsort(distances)[: self.k]
        # Get labels of the k nearest neighbors
        k_nearest_labels = self.y_train[k_indices]
        print(k_nearest_labels)
        # Majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        print(most_common)
        return most_common[0][0]


if __name__ == "__main__":
    # Training data (2D points and labels)
    X_train = [[1, 2], [2, 3], [3, 3], [6, 5], [7, 8], [8, 8]]
    y_train = [0, 0, 0, 1, 1, 1]
    # Test data
    X_test = [[4, 4], [7, 6]]

    # Instantiate k-NN with k=3
    k = 3
    knn = KNearestNeighbors(k=k)
    knn.fit(X_train, y_train)

    # Predict classes for test data
    predictions = knn.predict(X_test)
    print("Predictions:", predictions)
    X_class0 = np.array([x for x, y in zip(X_train, y_train) if y == 0])
    X_class1 = np.array([x for x, y in zip(X_train, y_train) if y == 1])
    X_test_class0 = np.array([x for x, y in zip(X_test, predictions) if y == 0])
    X_test_class1 = np.array([x for x, y in zip(X_test, predictions) if y == 1])
    X_class0 = np.vstack((X_class0, X_test_class0))
    X_class1 = np.vstack((X_class1, X_test_class1))

    plt.figure(figsize=(10, 6))
    plt.scatter(
        X_class0[:, 0], X_class0[:, 1], c="blue", edgecolor="k", label="Class 0"
    )
    plt.scatter(X_class1[:, 0], X_class1[:, 1], c="red", edgecolor="k", label="Class 1")
    plt.title("k-Nearest Neighbors Classification (k={0})".format(k))
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()
