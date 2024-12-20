import numpy as np


def simulate_random_walks(num_walks, num_steps):
    # Initialize arrays to store x and y positions for all walks
    x_positions = np.zeros((num_walks, num_steps + 1))
    y_positions = np.zeros((num_walks, num_steps + 1))

    # Simulate random steps for each walk
    for step in range(1, num_steps + 1):
        # Randomly choose -1, 0, or 1 for each direction for all walks
        x_steps = np.random.choice([-1, 1], size=num_walks)
        y_steps = np.random.choice([-1, 1], size=num_walks)

        # Update positions
        x_positions[:, step] = x_positions[:, step - 1] + x_steps
        y_positions[:, step] = y_positions[:, step - 1] + y_steps

    # Compute distances from the origin after num_steps
    final_distances = np.sqrt(x_positions[:, -1] ** 2 + y_positions[:, -1] ** 2)

    # Calculate average distance
    average_distance = np.mean(final_distances)

    return average_distance


if __name__ == "__main__":
    num_walks = 10000
    num_steps = 100
    average_distance = simulate_random_walks(num_walks, num_steps)
    print(
        f"Average distance from the origin after {num_steps} steps: {average_distance:.2f}"
    )
