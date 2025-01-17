import cv2
import matplotlib.pyplot as plt
import numpy as np


# Utility function to display images
def show_image(title, img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


# Load an example image
image = cv2.imread("lena.png")  # Replace with your image path
show_image("Original Image", image)


# Translation
rows, cols, _ = image.shape
transformation_matrix = np.float32([[1, 0, 50], [0, 1, 30]])
translated_image = cv2.warpAffine(image, transformation_matrix, (cols, rows))
show_image("Translated Image", translated_image)

# Rotation
center = (cols // 2, rows // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1)
rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
show_image("Rotated Image", rotated_image)

# Scaling
scaling_matrix = np.float32([[1.5, 0, 0], [0, 1.5, 0]])
scaled_image = cv2.warpAffine(image, scaling_matrix, (int(cols * 1.5), int(rows * 1.5)))
show_image("Scaled Image", scaled_image)
