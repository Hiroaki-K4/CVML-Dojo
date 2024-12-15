import numpy as np


def extract_odd_numbers(arr):
    odds = arr[arr % 2 != 0]
    return odds


if __name__ == "__main__":
    arr = np.array([1, 2, 3, 4, 5])
    odds = extract_odd_numbers(arr)
    print("Odd numbers: ", odds)
