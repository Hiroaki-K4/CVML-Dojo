import cv2
import numpy as np

image = cv2.imread("lena.png")
(h, w) = image.shape[:2]

# Define four points in the source image (corners of a quadrilateral)
source_points = np.array(
    [
        [100, 150],  # Top-left
        [400, 150],  # Top-right
        [50, 400],  # Bottom-left
        [450, 450],  # Bottom-right
    ],
    dtype=np.float32,
)

# Define the destination points (corners of the rectangle)
dest_points = np.array(
    [
        [0, 0],  # Top-left
        [300, 0],  # Top-right
        [0, 300],  # Bottom-left
        [300, 300],  # Bottom-right
    ],
    dtype=np.float32,
)

# Compute the perspective transformation matrix
M = cv2.getPerspectiveTransform(source_points, dest_points)

# Perform the perspective warp
warped_image = cv2.warpPerspective(image, M, (300, 300))

# Show the original and warped images
cv2.imshow("Original Image", image)
cv2.imshow("Warped Image", warped_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
