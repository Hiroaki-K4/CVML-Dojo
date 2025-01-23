import cv2
import matplotlib.pyplot as plt


def main():
    image_path = "lena.png"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in x-direction
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in y-direction

    # Normalize the gradients for display
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    # Display results
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1), plt.title("Original Image"), plt.axis("off")
    plt.imshow(image, cmap="gray")

    plt.subplot(2, 2, 2), plt.title("Sobel X"), plt.axis("off")
    plt.imshow(sobel_x, cmap="gray")

    plt.subplot(2, 2, 3), plt.title("Sobel Y"), plt.axis("off")
    plt.imshow(sobel_y, cmap="gray")

    plt.subplot(2, 2, 4), plt.title("Sobel Combined"), plt.axis("off")
    plt.imshow(sobel_combined, cmap="gray")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
