import numpy as np


def convolution(input_mat, kernel, pad_width=0, stride=1):
    kernel_rows, kernel_cols = kernel.shape

    pad_mat = np.pad(input_mat, pad_width=pad_width, mode="constant", constant_values=0)
    output_rows = ((pad_mat.shape[0] - kernel_rows) // stride) + 1
    output_cols = ((pad_mat.shape[1] - kernel_cols) // stride) + 1

    output = np.zeros((output_rows, output_cols))

    for i in range(output_rows):
        for j in range(output_cols):
            region = pad_mat[
                i * stride : i * stride + kernel_rows,
                j * stride : j * stride + kernel_cols,
            ]
            output[i, j] = np.sum(region * kernel)

    return output


if __name__ == "__main__":
    input_mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    kernel = np.array([[1, 0], [0, -1]])
    print("Input: ")
    print(input_mat)
    print()

    pad_width = 0
    stride = 1
    print("Pad width: {0}, stride={1}".format(pad_width, stride))
    print(convolution(input_mat, kernel, pad_width, stride))
    print()

    pad_width = 1
    stride = 1
    print("Pad width: {0}, stride={1}".format(pad_width, stride))
    print(convolution(input_mat, kernel, pad_width, stride))
    print()

    pad_width = 1
    stride = 2
    print("Pad width: {0}, stride={1}".format(pad_width, stride))
    print(convolution(input_mat, kernel, pad_width, stride))
    print()
