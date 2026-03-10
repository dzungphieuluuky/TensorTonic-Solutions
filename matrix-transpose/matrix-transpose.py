import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.array(A)
    width, height = A.shape
    B = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            B[i][j] = A[j][i]
    return B
