import numpy as np


def flatten_image(img):
    flatten_img = np.reshape(img, (-1, 3))
    return flatten_img


if __name__ == "__main__":
    img = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])
    print("Input {0}:".format(img.shape))
    print(img)
    print()

    flatten_img = flatten_image(img)
    print("Output {0}:".format(flatten_img.shape))
    print(flatten_img)
