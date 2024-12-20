import numpy as np

# def compute_sums(arr):
#     result = []
#     for i in range(len(arr)):
#         for j in range(len(arr[0])):
#             result.append(arr[i, j] + i + j)
# return np.array(result).reshape(arr.shape)


def compute_sums(arr):
    rows, cols = arr.shape
    row_indices = np.arange(rows).reshape(-1, 1)  # Column vector for row indices
    col_indices = np.arange(cols).reshape(1, -1)  # Row vector for column indices

    return arr + row_indices + col_indices


if __name__ == "__main__":
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("Input:")
    print(arr)
    print()

    print("Output:")
    print(compute_sums(arr))
