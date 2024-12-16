import numpy as np


def find_unique_rows(mat):
    return np.unique(mat, axis=0)


if __name__ == "__main__":
    mat = np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6], [7, 8, 9], [7, 8, 9]])
    print("Input: ", mat)
    print("Result: ", find_unique_rows(mat))
