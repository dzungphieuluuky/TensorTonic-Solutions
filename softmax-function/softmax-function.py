import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x = np.array(x)
    axes = x.ndim - 1
    shifted = x - np.max(x, axis=axes, keepdims=True)
    sum_exp = np.sum(np.exp(shifted), axis=axes, keepdims=True)
    return np.exp(shifted)/sum_exp