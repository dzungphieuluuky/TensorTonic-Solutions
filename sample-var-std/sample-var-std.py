import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    # Write code here
    x = np.asarray(x)
    var = np.var(x)
    normalized_var = var * len(x) / (len(x) - 1)
    return normalized_var, np.sqrt(normalized_var)