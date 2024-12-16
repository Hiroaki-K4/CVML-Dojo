import numpy as np


def diagonal_product(mat):
    n = mat.shape[0]
    primary_product = 1
    secondary_product = 1
    for i in range(n):
        primary_product *= mat[i, i]
        secondary_product *= mat[i, n - i - 1]

    return primary_product, secondary_product


if __name__ == "__main__":
    mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    primary_product, secondary_product = diagonal_product(mat)
    print("Input: ")
    print(mat)
    print(
        "Primary product: {0}, Secondary product: {1}".format(
            primary_product, secondary_product
        )
    )
