import cv2
import matplotlib.pyplot as plt

# Load images
img1 = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)  # Query image
img2 = cv2.imread("lena_rotate.png", cv2.IMREAD_GRAYSCALE)  # Train image

# Step 1: Detect ORB keypoints and compute descriptors
orb = cv2.ORB_create()

# Detect and compute keypoints and descriptors
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

# Step 2: Match features using Brute-Force Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(descriptors1, descriptors2)

# Sort matches by distance (smaller distance = better match)
matches = sorted(matches, key=lambda x: x.distance)

# Step 3: Visualize the matches
# Draw the top 20 matches
img_matches = cv2.drawMatches(
    img1,
    keypoints1,
    img2,
    keypoints2,
    matches[:20],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

# Display the matches
plt.figure(figsize=(15, 10))
plt.imshow(img_matches)
plt.title("Feature Matching with ORB")
plt.show()
