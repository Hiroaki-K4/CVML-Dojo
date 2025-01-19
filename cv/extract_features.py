import cv2
import matplotlib.pyplot as plt


def orb_feature_extraction(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors


def sift_feature_extraction(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


def visualize_keypoints(image, keypoints, title="Keypoints"):
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))
    plt.figure(figsize=(8, 6))
    plt.imshow(image_with_keypoints, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    image_path = "lena.png"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # ORB Example
    orb_kp, orb_desc = orb_feature_extraction(image)
    print(
        f"ORB: Found {len(orb_kp)} keypoints and descriptors with shape {orb_desc.shape}"
    )
    visualize_keypoints(image, orb_kp, "ORB Keypoints")

    # SIFT Example
    sift_kp, sift_desc = sift_feature_extraction(image)
    print(
        f"SIFT: Found {len(sift_kp)} keypoints and descriptors with shape {sift_desc.shape}"
    )
    visualize_keypoints(image, sift_kp, "SIFT Keypoints")
