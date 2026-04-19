import numpy as np

def make_diagonal(v):
    """
    Returns: (n, n) NumPy array with v on the main diagonal
    """
    # Write code here
    v = np.array(v)
    return v.reshape(1, -1) * np.ones(v.shape) * np.eye(v.shape[0])
