import cv2
import matplotlib.pyplot as plt


def plot_histograms(original, equalized):
    plt.figure(figsize=(12, 6))

    # Original image histogram
    plt.subplot(2, 2, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.hist(original.ravel(), bins=256, range=[0, 256], color="blue", alpha=0.7)
    plt.title("Original Histogram")

    # Equalized image histogram
    plt.subplot(2, 2, 3)
    plt.imshow(equalized, cmap="gray")
    plt.title("Equalized Image")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.hist(equalized.ravel(), bins=256, range=[0, 256], color="green", alpha=0.7)
    plt.title("Equalized Histogram")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)
    equalized_image = cv2.equalizeHist(image)
    plot_histograms(image, equalized_image)
