import numpy as np


def moving_average(arr, window_size):
    if window_size <= 0 or window_size > arr.shape[0]:
        raise RuntimeError("Window size is incorrect.")
    cumsum = np.cumsum(arr)
    cumsum = np.insert(cumsum, 0, 0)
    cumsum_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    return cumsum_avg


if __name__ == "__main__":
    arr = np.array([1, 2, 3, 4, 5, 6])
    window_size = 3
    print("Input array: {0}, window size: {1}".format(arr, window_size))
    print("Moving average: ", moving_average(arr, window_size))
