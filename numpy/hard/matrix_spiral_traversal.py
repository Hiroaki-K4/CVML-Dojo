import numpy as np


def matrix_spiral_traversal(matrix):
    result = []
    if matrix.size == 0:
        return result

    top, bottom = 0, matrix.shape[0] - 1
    left, right = 0, matrix.shape[1] - 1

    while top <= bottom and left <= right:
        # Traverse from left to right across the top row
        result.extend(matrix[top, left : right + 1].tolist())
        top += 1

        # Traverse from top to bottom down the right column
        if top <= bottom:
            result.extend(matrix[top : bottom + 1, right].tolist())
            right -= 1

        # Traverse from right to left across the bottom row
        if top <= bottom:
            result.extend(matrix[bottom, left : right + 1][::-1].tolist())
            bottom -= 1

        # Traverse from bottom to top up the left column
        if left <= right:
            # result.extend(matrix[bottom:top - 1:-1, left].tolist())
            result.extend(matrix[top : bottom + 1, left][::-1].tolist())
            left += 1

    return result


if __name__ == "__main__":
    mat = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    print(matrix_spiral_traversal(mat))
