import cv2
import numpy as np


def main():
    # Define the camera matrix (intrinsic parameters)
    # Replace these values with your actual camera parameters
    K = np.array(
        [
            [1000, 0, 640],  # fx, 0, cx
            [0, 1000, 360],  # 0, fy, cy
            [0, 0, 1],  # 0,  0,  1
        ]
    )

    # Example homography matrix (replace with actual computation)
    H = np.array([[1.2, 0.2, 100], [0.1, 1.1, 200], [0.001, 0.002, 1]])

    # Decompose the homography into rotation, translation, and normal vectors
    # Returns multiple solutions
    _, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K)

    # Print the results
    for i, (R, T, N) in enumerate(zip(Rs, Ts, Ns)):
        print(f"Solution {i + 1}:")
        print("Rotation Matrix (R):\n", R)
        print("Translation Vector (T):\n", T)
        print("Plane Normal Vector (N):\n", N)
        print()

    # Verifying properties (optional)
    print("Homography decomposition yields up to 4 solutions.")


if __name__ == "__main__":
    main()
