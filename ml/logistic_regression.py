import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Number of samples and features
        num_samples, num_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.iterations):
            # Linear model
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)

            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        # Linear model
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        # Convert probabilities to binary classification
        return np.where(y_pred >= 0.5, 1, 0)


# class LogisticRegression:


# Example usage:
if __name__ == "__main__":
    # Sample dataset (X: features, y: labels)
    X = np.array([[0, 1], [1, 1], [2, 1], [3, 1]])
    y = np.array([0, 0, 1, 1])

    # Train logistic regression model
    model = LogisticRegression(learning_rate=0.1, iterations=1000)
    model.fit(X, y)

    # Predict
    predictions = model.predict(X)
    print("Correct labels: ", y)
    print("Predictions:", predictions)
