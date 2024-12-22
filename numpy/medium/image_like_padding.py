import numpy as np


def image_like_padding():
    random_mat = np.random.randint(low=1, high=11, size=(5, 5))
    pad_mat = np.pad(random_mat, pad_width=1, mode="constant", constant_values=0)
    return pad_mat


if __name__ == "__main__":
    print(image_like_padding())
