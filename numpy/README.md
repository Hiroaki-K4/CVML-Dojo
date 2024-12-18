# Numpy

## Easy
1. **Matrix Initialization**  
Create a NumPy array of shape (5, 5) where the border elements are 1 and the inner elements are 0.
```
[[1 1 1 1 1]
 [1 0 0 0 1]
 [1 0 0 0 1]
 [1 0 0 0 1]
 [1 1 1 1 1]]
```

2. **Row and Column Sums**  
Given a matrix, return two arrays: one containing the sum of each row and the other containing the sum of each column.
3. **Extract Odd Numbers**  
Write a function that extracts all odd numbers from a 1D NumPy array and returns them in a new array.
4. **Replace Values**  
Replace all negative numbers in a 2D array with 0 without creating a new array.

## Medium
1. **Matrix Multiplication**  
Implement a function that takes two 2D matrices and performs matrix multiplication without using np.dot or np.matmul.
2. **Image-Like Padding**  
Create a 2D array of shape (5, 5) with random integers between 1 and 10. Then, pad the array with zeros so that the final shape is (7, 7).
3. **Find Unique Rows**  
Write a function that identifies and returns only the unique rows from a 2D array.
4. **Moving Average**  
Implement a function to calculate the moving average of a 1D NumPy array with a given window size.
5. **Diagonal Product**  
Given a square matrix, calculate the product of the elements along its primary diagonal and secondary diagonal.

## Hard
1. **Convolution**  
Implement a function to perform a 2D convolution between an input matrix and a kernel without using any external libraries.
2. **Nearest Neighbor Interpolation**  
Given a 2D array, implement nearest-neighbor interpolation to resize it to a new shape.
3. **Matrix Spiral Traversal**  
Write a function that traverses a 2D matrix in a spiral order and returns the elements as a 1D array.
4. **Count Subarrays with Target Sum**  
Given a 1D array, count the number of contiguous subarrays that sum up to a given target using efficient array slicing techniques.
5. **Custom Sorting**  
Given a 2D array, sort all rows based on the values in the last column.

## Performance and Optimization Challenges
1. **Optimize Nested Loops**  
Rewrite a given Python function with nested loops into a fully vectorized NumPy implementation.

```python
def compute_sums(arr):
    result = []
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            result.append(arr[i, j] + i + j)
    return np.array(result).reshape(arr.shape)
```

2. **Efficient Distance Calculation**  
Given two 2D arrays representing points in 2D space, compute the Euclidean distances between each pair of points without using loops.
3. **Simulate Random Walks**  
Write a function that simulates multiple random walks in a 2D grid and calculates the average distance from the origin after n steps.

## Practical Problems
1. **Flatten an Image**  
Given a 3D array representing an RGB image, flatten it into a 2D array where each row represents the RGB values of a pixel.
2. **Histogram Equalization**  
Implement histogram equalization for a grayscale image represented as a 2D NumPy array.
3. **Find Missing Data**  
Given a 2D array with some missing values represented as np.nan, replace the missing values with the mean of their respective columns.
4. **Stock Price Analysis**  
Given a 1D array of daily stock prices, calculate the maximum profit that can be made by buying and selling the stock once.
