import numpy as np


def sum_row_and_column(mat):
    row_sum = np.sum(mat, axis=1)
    col_sum = np.sum(mat, axis=0)

    return row_sum, col_sum


if __name__ == "__main__":
    mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    row_sum, col_sum = sum_row_and_column(mat)
    print("Row sum: ", row_sum)
    print("Column sum: ", col_sum)
