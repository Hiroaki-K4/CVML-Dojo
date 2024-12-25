import matplotlib.pyplot as plt
import numpy as np


def he_initialization(input_size, output_size):
    """He initialization: N(0, sqrt(2 / input_size))"""
    std = np.sqrt(2 / input_size)
    return np.random.normal(0, std, (input_size, output_size))


def xavier_initialization(input_size, output_size):
    """Xavier initialization: N(0, sqrt(1 / input_size))"""
    std = np.sqrt(1 / input_size)
    return np.random.normal(0, std, (input_size, output_size))


def visualize_initialization(init_func, input_size, output_size, num_samples=10000):
    """Visualize weight distributions for a given initialization method."""
    weights = [init_func(input_size, output_size).flatten() for _ in range(num_samples)]
    weights = np.concatenate(weights)

    plt.hist(weights, bins=50, alpha=0.75, label=init_func.__name__)
    plt.title(f"{init_func.__name__.replace('_', ' ').capitalize()} Distribution")
    plt.xlabel("Weight Values")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Parameters for the visualizations
    input_size = 100
    output_size = 50

    # Visualize He Initialization
    visualize_initialization(he_initialization, input_size, output_size)

    # Visualize Xavier Initialization
    visualize_initialization(xavier_initialization, input_size, output_size)
