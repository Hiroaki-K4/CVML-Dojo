import numpy as np


def pairwise_distances(arr1, arr2):
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    # Compute pairwise differences
    differences = arr1[:, np.newaxis, :] - arr2[np.newaxis, :, :]  # Shape: (m, n, 2)
    # Compute squared distances and then square root
    distances = np.sqrt(np.sum(differences ** 2, axis=-1))  # Shape: (m, n)

    return distances


if __name__ == "__main__":
    arr1 = np.array([[1, 2], [3, 4], [5, 6]])
    arr2 = np.array([[7, 8], [9, 10]])
    distances = pairwise_distances(arr1, arr2)
    print(distances)
