import numpy as np


def nearest_neighbor_interpolation(arr, new_shape):
    old_rows, old_cols = arr.shape
    new_rows, new_cols = new_shape

    row_scale = old_rows / new_rows
    col_scale = old_cols / new_cols

    resized = np.zeros(new_shape, dtype="int")
    for i in range(new_rows):
        for j in range(new_cols):
            resized[i, j] = arr[int(i * row_scale), int(j * col_scale)]

    return resized


if __name__ == "__main__":
    arr = np.array([[1, 2], [3, 4]])
    print("Input:")
    print(arr)
    print()

    new_shape = (4, 6)
    print("Shape {0}:".format(new_shape))
    print(nearest_neighbor_interpolation(arr, new_shape))
    print()

    new_shape = (2, 2)
    print("Shape {0}:".format(new_shape))
    print(nearest_neighbor_interpolation(arr, new_shape))
