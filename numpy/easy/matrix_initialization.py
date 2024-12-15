import numpy as np


def initialize_matrix():
    ans_mat = np.zeros((5, 5), dtype=int)
    ans_mat[0, :] = 1
    ans_mat[-1, :] = 1
    ans_mat[:, 0] = 1
    ans_mat[:, -1] = 1

    return ans_mat


if __name__ == "__main__":
    print(initialize_matrix())
