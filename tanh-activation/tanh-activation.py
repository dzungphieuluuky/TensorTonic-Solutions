import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x = np.asarray(x)
    return (np.exp(2*x) - 1) / (np.exp(2*x) + 1)