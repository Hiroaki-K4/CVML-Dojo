import numpy as np


def pooling_2d(input_mat, pool_size, stride, mode="max"):
    input_height, input_width, channels = input_mat.shape
    output_height = (input_height - pool_size) // stride + 1
    output_width = (input_width - pool_size) // stride + 1

    pool_mat = np.zeros((output_height, output_width, channels))

    for h in range(output_height):
        for w in range(output_width):
            for c in range(channels):
                h_start = h * stride
                h_end = h_start + pool_size
                w_start = w * stride
                w_end = w_start + pool_size

                window = input_mat[h_start:h_end, w_start:w_end, c]
                if mode == "max":
                    pool_mat[h, w, c] = np.max(window)
                elif mode == "average":
                    pool_mat[h, w, c] = np.mean(window)
                elif mode == "min":
                    pool_mat[h, w, c] = np.min(window)
                else:
                    raise ValueError("Mode is invalid")

    return pool_mat


if __name__ == "__main__":
    input_data = np.random.rand(6, 6, 1)
    print("Input Data:\n", input_data[:, :, 0])
    print()
    pooled_output = pooling_2d(input_data, pool_size=2, stride=2, mode="max")
    print("Pooled Output:\n", pooled_output[:, :, 0])
