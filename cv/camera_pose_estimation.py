import cv2
import numpy as np

# Step 1: Define 3D points in the object coordinate space (e.g., a cube)
object_points = np.array(
    [
        [0.0, 0.0, 0.0],  # Point 1
        [1.0, 0.0, 0.0],  # Point 2
        [1.0, 1.0, 0.0],  # Point 3
        [0.0, 1.0, 0.0],  # Point 4
        [0.0, 0.0, 1.0],  # Point 5
        [1.0, 0.0, 1.0],  # Point 6
        [1.0, 1.0, 1.0],  # Point 7
        [0.0, 1.0, 1.0],  # Point 8
    ],
    dtype=np.float32,
)

# Step 2: Define corresponding 2D points in the image plane
image_points = np.array(
    [
        [322, 240],  # Projection of Point 1
        [450, 240],  # Projection of Point 2
        [450, 360],  # Projection of Point 3
        [322, 360],  # Projection of Point 4
        [300, 200],  # Projection of Point 5
        [430, 200],  # Projection of Point 6
        [430, 320],  # Projection of Point 7
        [300, 320],  # Projection of Point 8
    ],
    dtype=np.float32,
)

# Step 3: Define the camera matrix (intrinsic parameters)
# Assuming fx = fy = 800, cx = 320, cy = 240, no skew
camera_matrix = np.array(
    [
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1],
    ],
    dtype=np.float64,
)

# Step 4: Define distortion coefficients (assuming zero distortion for simplicity)
dist_coeffs = np.zeros((4, 1))  # [k1, k2, p1, p2]

# Step 5: Solve the PnP problem to estimate the camera pose
success, rvec, tvec = cv2.solvePnP(
    object_points, image_points, camera_matrix, dist_coeffs
)

if success:
    print("Rotation Vector (rvec):\n", rvec)
    print("Translation Vector (tvec):\n", tvec)

    # Convert rvec to a rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    print("Rotation Matrix:\n", rotation_matrix)

    # Camera position in world coordinates
    camera_position = -np.linalg.inv(rotation_matrix) @ tvec
    print("Camera Position:\n", camera_position)
else:
    print("Pose estimation failed.")
