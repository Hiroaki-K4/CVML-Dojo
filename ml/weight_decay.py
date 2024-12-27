import numpy as np


def weight_decay():
    # Dummy dataset
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([7, 10, 13])

    # Initialize weights and hyperparameters
    weights = np.random.randn(2)
    learning_rate = 0.01
    weight_decay = 0.001  # λ

    # Number of iterations
    num_epochs = 1000

    # Gradient descent with weight decay
    for epoch in range(num_epochs):
        # Predict
        predictions = np.dot(X, weights)
        # Compute the error
        error = predictions - y
        # Compute gradients (mean squared error gradient)
        gradient = (2 / len(y)) * np.dot(X.T, error)
        # Apply weight decay
        weights = weights * (1 - learning_rate * weight_decay)  # Decay term
        weights -= learning_rate * gradient  # Gradient descent step
        # Print loss every 100 epochs
        if epoch % 100 == 0:
            loss = np.mean(error ** 2)
            print(f"Epoch {epoch}, Loss: {loss}")

    # Final weights
    print("Trained weights:", weights)


if __name__ == "__main__":
    weight_decay()
