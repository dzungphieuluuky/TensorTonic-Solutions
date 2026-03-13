import numpy as np

def sigmoid(x):
    x = np.array(x)
    deno = 1 + np.exp(-x)
    return 1 / deno
def swish(x):
    """
    Implement Swish activation function.
    """
    x = np.array(x)
    return x * sigmoid(x)