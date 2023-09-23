import numpy as np
from scipy.interpolate import NearestNDInterpolator


def nearest_zero(matrix):
    # Get the indices of the 0 values in the matrix
    zero_indices = np.argwhere(matrix == 0)

    # Get the indices of the non-zero values in the matrix
    non_zero_indices = np.argwhere(matrix != 0)

    # Get the non-zero values in the matrix
    non_zero_values = matrix[non_zero_indices[:, 0], non_zero_indices[:, 1]]

    # Create a nearest-neighbor interpolator using the non-zero values
    interpolator = NearestNDInterpolator(non_zero_indices, non_zero_values)

    for i, j in zero_indices:
        if i > 0 and i < matrix.shape[0] - 1 and j > 0 and j < matrix.shape[1] - 1:
            if (matrix[i - 1, j] != 0 and matrix[i + 1, j]) != 0 or (matrix[i, j - 1] != 0 and matrix[i, j + 1] != 0):
                matrix[i, j] = interpolator([i, j])

    return matrix
