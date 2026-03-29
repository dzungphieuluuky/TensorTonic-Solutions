import numpy as np

def make_diagonal(v):
    """
    Returns: (n, n) NumPy array with v on the main diagonal
    """
    n = len(v)
    res = np.zeros((n, n))
    for i in range(n):
        res[i][i] = v[i]
    return res
