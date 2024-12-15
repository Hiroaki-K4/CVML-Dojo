import numpy as np


def replace_values(mat):
    mat[mat < 0] = 0


if __name__ == "__main__":
    mat = np.array([[-1, 2, 3], [4, -5, 6], [7, 8, -9]])
    replace_values(mat)
    print("Replaced values")
    print(mat)
