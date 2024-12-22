import numpy as np

# Generate a synthetic dataset
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # Features
y = 4 + 3 * X + np.random.randn(100, 1)  # Labels with noise

# Add a bias term to X (for the intercept)
X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add a column of 1s to X

# Initialize parameters (weights and bias)
theta = np.random.randn(2, 1)  # Random initialization

# Define hyperparameters
learning_rate = 0.1
n_iterations = 1000
m = X_b.shape[0]  # Number of samples

# Gradient Descent
for iteration in range(n_iterations):
    gradients = 1 / m * X_b.T @ (X_b @ theta - y)  # Compute gradients
    theta -= learning_rate * gradients  # Update parameters

# Display the final parameters
print("Learned parameters (theta):")
print(theta)

# Predictions
y_pred = X_b @ theta

# Visualization (optional)
import matplotlib.pyplot as plt

plt.scatter(X, y, label="Data")
plt.plot(X, y_pred, color="red", label="Regression Line")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
