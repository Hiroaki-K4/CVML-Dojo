import numpy as np


def mean_squared_error(y, t):
    return 0.5 * np.sum((np.array(y) - np.array(t)) ** 2)


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(np.array(t) * np.log(np.array(y) + delta))


if __name__ == "__main__":
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    print("MSE: ", mean_squared_error(y, t))
    print("Cross entropy error: ", cross_entropy_error(y, t))
    y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    print("MSE: ", mean_squared_error(y, t))
    print("Cross entropy error: ", cross_entropy_error(y, t))
