import numpy as np


def matrix_manipulation(mat1, mat2):
    if mat1.shape[1] != mat2.shape[0] or mat1.shape[0] != mat2.shape[1]:
        raise RuntimeError("Row and columns of two matrix are invalid!!")

    result = np.zeros((mat1.shape[0], mat2.shape[1]), dtype=int)
    for i in range(mat1.shape[0]):
        for j in range(mat2.shape[1]):
            for k in range(mat1.shape[1]):
                result[i, j] += mat1[i, k] * mat2[k, j]

    return result


if __name__ == "__main__":
    mat1 = np.array([[1, 2, 3], [4, 5, 6]])
    mat2 = np.array([[1, 2], [3, 4], [5, 6]])
    print("Numpy dot: ", np.dot(mat1, mat2))
    print("Original: ", matrix_manipulation(mat1, mat2))
