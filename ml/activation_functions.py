import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def softmax(x):
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=0)


if __name__ == "__main__":
    x = np.array([-1, 0, 1, 2, -3])
    print("Sigmoid: ", sigmoid(x))
    print("Relu: ", relu(x))
    print("Leaky relu: ", leaky_relu(x))
    x = np.array([2.0, 1.0, 0.1])
    print("Softmax: ", softmax(x))
