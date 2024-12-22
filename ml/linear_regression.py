import matplotlib.pyplot as plt
import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None

    def fit(self, X, y):
        # Add a bias term to X (for the intercept)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add a column of 1s to X
        self.theta = np.random.randn(2, 1)  # Random initialization
        m = X_b.shape[0]  # Number of samples
        for iteration in range(self.n_iterations):
            gradients = 1 / m * X_b.T @ (X_b @ self.theta - y)  # Compute gradients
            self.theta -= self.learning_rate * gradients  # Update parameters
        print("Learned parameters (theta):")
        print(self.theta)

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.theta


if __name__ == "__main__":
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)  # Features
    y = 4 + 3 * X + np.random.randn(100, 1)  # Labels with noise
    learning_rate = 0.1
    n_iterations = 1000
    model = LinearRegression(learning_rate, n_iterations)
    model.fit(X, y)
    y_pred = model.predict(X)

    plt.scatter(X, y, label="Data")
    plt.plot(X, y_pred, color="red", label="Regression Line")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()
