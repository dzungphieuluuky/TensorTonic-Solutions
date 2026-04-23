import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    # Write code here
    array_x = np.array(x)
    if array_x.ndim != 3 and array_x.ndim != 4:
        raise ValueError
    return np.mean(array_x, axis=(-1, -2))