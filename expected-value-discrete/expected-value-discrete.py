import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x = np.array(x)
    p = np.array(p)
    if x.shape != p.shape:
        raise ValueError
    if np.sum(p) != 1 :
        raise ValueError
    return np.sum(x * p)
