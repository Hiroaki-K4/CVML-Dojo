import cv2
import matplotlib.pyplot as plt


def main():
    image_path = "lena.png"  # Replace with the path to your image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 1)

    # Apply Canny Edge Detection
    low_threshold = 50
    high_threshold = 150
    edges_canny = cv2.Canny(blurred, low_threshold, high_threshold)

    # Display results
    plt.figure(figsize=(6, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Canny Edge Detection")
    plt.imshow(edges_canny, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
