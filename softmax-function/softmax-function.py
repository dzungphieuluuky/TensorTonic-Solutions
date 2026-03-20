import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    x = np.array(x)
    if x.ndim == 1:
        max_logit = np.max(x)
        exp_logit = np.exp(x - max_logit)
        return exp_logit / np.sum(exp_logit)
    elif x.ndim == 2:
        max_logit = np.max(x, axis=1, keepdims=True)
        exp_logit = np.exp(x - max_logit)
        return exp_logit / np.sum(exp_logit, axis=1, keepdims=True)