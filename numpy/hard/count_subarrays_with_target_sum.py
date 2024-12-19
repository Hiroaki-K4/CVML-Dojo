import numpy as np


def count_subarrays_with_target_sum(arr, target):
    arr = np.array(arr)  # Ensure the input is a NumPy array
    prefix_sums = {0: 1}  # Initialize with 0 sum having a frequency of 1
    current_sum = 0
    count = 0

    for num in arr:
        current_sum += num
        complement = current_sum - target
        count += prefix_sums.get(
            complement, 0
        )  # Add frequency of the complement if it exists
        prefix_sums[current_sum] = prefix_sums.get(current_sum, 0) + 1

    return count


if __name__ == "__main__":
    arr = np.array([-1, 1, 1, 2, 3, 1, 2, 2])
    target = 4
    print("Input: {0}, Target: {1}".format(arr, target))
    print("Output: ", count_subarrays_with_target_sum(arr, target))
