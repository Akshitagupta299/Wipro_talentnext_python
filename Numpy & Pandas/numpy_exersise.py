# 1.   Create two dimensional 3*3 array and perform ndim, shape, slicing operation on it
import numpy as np

# Create a 3x3 array
arr2D = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

print("2D Array:\n", arr2D)

# ndim → number of dimensions
print("Number of dimensions:", arr2D.ndim)

# shape → rows and columns
print("Shape of array:", arr2D.shape)

# Slicing → accessing parts of the array
print("First row:", arr2D[0, :])       # all columns of first row
print("Second column:", arr2D[:, 1])   # all rows of second column
print("Element at [2,2]:", arr2D[1,1]) # element at row 2, col 2

# 2.  Create one dimensional array and perform ndim, shape, reshape operation on it.
# Create a 1D array
arr1D = np.array([10, 20, 30, 40, 50, 60])

print("1D Array:", arr1D)

# ndim
print("Number of dimensions:", arr1D.ndim)

# shape
print("Shape of array:", arr1D.shape)

# reshape (convert 1D → 2D e.g. 2x3)
reshaped_arr = arr1D.reshape(2, 3)
print("Reshaped Array (2x3):\n", reshaped_arr)
