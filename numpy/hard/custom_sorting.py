import numpy as np


def sort_rows_by_last_column(matrix):
    matrix = np.array(matrix)
    sorted_matrix = matrix[matrix[:, -1].argsort(), :]
    return sorted_matrix


if __name__ == "__main__":
    matrix = np.array([[1, 3, 5], [4, 2, 1], [7, 8, 0], [6, 5, 9]])

    sorted_matrix = sort_rows_by_last_column(matrix)
    print("Sorted Matrix:")
    print(sorted_matrix)
