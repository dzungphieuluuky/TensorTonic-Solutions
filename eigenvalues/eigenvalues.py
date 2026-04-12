import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    # Write code here
    if not matrix:
        return None
    if not isinstance(matrix[0], list):
        return None
    n, m = len(matrix), len(matrix[0])
    for i in range(n):
        if len(matrix[i]) != m:
            return None
    matrix = np.asarray(matrix)
    if matrix.shape[0] != matrix.shape[1]:
        return None
    eigens = np.linalg.eigvals(matrix)
    np.lexsort(eigens)
    return eigens