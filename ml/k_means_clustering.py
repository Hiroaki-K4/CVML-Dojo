import matplotlib.pyplot as plt
import numpy as np


class KMeansClustering:
    def __init__(self, data, k, max_iters=100, tolerance=1e-4):
        self.data = data
        self.k = k
        self.max_iters = max_iters
        self.tolerance = tolerance

    def initialize_centroids(self):
        """Randomly initialize k centroids from the data."""
        indices = np.random.choice(self.data.shape[0], self.k, replace=False)
        return self.data[indices]

    def assign_clusters(self, centroids):
        """Assign each data point to the nearest centroid."""
        distances = np.linalg.norm(self.data[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    def update_centroids(self, labels):
        """Recalculate centroids as the mean of the points in each cluster."""
        centroids = np.array(
            [self.data[labels == i].mean(axis=0) for i in range(self.k)]
        )
        return centroids

    def k_means(self):
        """K-means clustering algorithm."""
        # Step 1: Initialize centroids
        centroids = self.initialize_centroids()

        for iteration in range(self.max_iters):
            # Step 2: Assign clusters
            labels = self.assign_clusters(centroids)

            # Step 3: Calculate new centroids
            new_centroids = self.update_centroids(labels)

            # Step 4: Check for convergence
            if np.all(np.abs(new_centroids - centroids) < self.tolerance):
                print(f"Converged after {iteration + 1} iterations.")
                break

            centroids = new_centroids

        return labels, centroids


if __name__ == "__main__":
    data = np.array(
        [
            [1.0, 2.0],
            [1.5, 1.8],
            [5.0, 8.0],
            [8.0, 8.0],
            [1.0, 0.6],
            [9.0, 11.0],
            [8.0, 2.0],
            [10.0, 2.0],
            [9.0, 3.0],
        ]
    )

    k = 3
    kmean = KMeansClustering(data, k)
    labels, centroids = kmean.k_means()
    print("Final centroids:", centroids)
    print("Cluster assignments:", labels)

    np.random.seed(42)
    colors = np.random.rand(k, 3)
    for i in range(k):
        X = np.array([x for x, y in zip(data, labels) if y == i])
        if X.shape[0] > 0:
            plt.scatter(X[:, 0], X[:, 1], color=colors[i])

    plt.show()
