import numpy as np


class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.learning_rate * grads[key]


class Momentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}

    def update(self, params, grads):
        for key in params.keys():
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])
            self.velocity[key] = (
                self.momentum * self.velocity[key] - self.learning_rate * grads[key]
            )
            params[key] += self.velocity[key]


class AdaGrad:
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.accumulated_grads = {}

    def update(self, params, grads):
        for key in params.keys():
            if key not in self.accumulated_grads:
                self.accumulated_grads[key] = np.zeros_like(params[key])
            self.accumulated_grads[key] += grads[key] ** 2
            params[key] -= (self.learning_rate * grads[key]) / (
                np.sqrt(self.accumulated_grads[key] + self.epsilon)
            )


# Adam


if __name__ == "__main__":
    params = {"w": np.random.randn(10), "b": np.random.randn(1)}
    grads = {"w": np.random.randn(10), "b": np.random.randn(1)}
    optimizer = SGD(learning_rate=0.01)
    print("Before optimization: ", params)
    optimizer.update(params, grads)
    print("After optimization(SGD): ", params)
    print()

    params = {"w": np.random.randn(10), "b": np.random.randn(1)}
    grads = {"w": np.random.randn(10), "b": np.random.randn(1)}
    optimizer = Momentum(learning_rate=0.01)
    print("Before optimization: ", params)
    optimizer.update(params, grads)
    print("After optimization(Momentum): ", params)
    print()

    params = {"w": np.random.randn(10), "b": np.random.randn(1)}
    grads = {"w": np.random.randn(10), "b": np.random.randn(1)}
    optimizer = AdaGrad(learning_rate=0.01)
    print("Before optimization: ", params)
    optimizer.update(params, grads)
    print("After optimization(AdaGrad): ", params)
    print()
