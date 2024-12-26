import numpy as np


def batch_normalization(x, gamma, beta, epsilon=1e-8):
    # 1. Calculate the mean and variance along the batch dimension (axis 0)
    mean = np.mean(x, axis=0)
    variance = np.var(x, axis=0)
    # 2. Normalize the input
    x_norm = (x - mean) / np.sqrt(variance + epsilon)
    # 3. Scale and shift the normalized input
    x_out = gamma * x_norm + beta

    return x_out


if __name__ == "__main__":
    batch_size = 32
    features = 100

    # Generate some random input data
    x = np.random.randn(batch_size, features)

    # Initialize gamma and beta (these are learnable parameters in a neural network)
    gamma = np.ones(features)  # Initialize with ones
    beta = np.zeros(features)  # Initialize with zeros

    # Perform batch normalization
    x_bn = batch_normalization(x, gamma, beta)

    # Print shapes to verify the output
    print("Input shape:", x.shape)
    print("Output shape:", x_bn.shape)

    # Demonstrating the effect of normalization (mean close to 0, std close to 1)
    print("Mean of normalized data along features:", np.mean(x_bn, axis=0))
    print("Std of normalized data along features:", np.std(x_bn, axis=0))

    # Example showing how gamma and beta affect the output
    gamma_2 = np.full(features, 2.0)
    beta_5 = np.full(features, 5.0)
    x_bn_2 = batch_normalization(x, gamma_2, beta_5)

    print("Mean of normalized data with gamma=2, beta=5:", np.mean(x_bn_2, axis=0))
    print("Std of normalized data with gamma=2, beta=5:", np.std(x_bn_2, axis=0))
