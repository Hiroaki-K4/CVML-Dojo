import cv2
import numpy as np

# Load an image
image = cv2.imread("lena.png")

# Display the original image
cv2.imshow("Original Image", image)

# 1. Resize the image
# Resize the image to half its original dimensions
height, width = image.shape[:2]
new_width = width // 2
new_height = height // 2
resized_image = cv2.resize(
    image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
)
cv2.imshow("Resized Image (Half Size)", resized_image)

# Resize the image to double its original dimensions
doubled_image = cv2.resize(
    image, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC
)
cv2.imshow("Resized Image (Double Size)", doubled_image)

# 2. Apply a filter using cv2.filter2D
# Define a sharpening kernel
sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

# Apply the filter
sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)
cv2.imshow("Sharpened Image", sharpened_image)

# Define a blur kernel
blur_kernel = np.ones((5, 5), np.float32) / 25

# Apply the blur filter
blurred_image = cv2.filter2D(image, -1, blur_kernel)
cv2.imshow("Blurred Image", blurred_image)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
