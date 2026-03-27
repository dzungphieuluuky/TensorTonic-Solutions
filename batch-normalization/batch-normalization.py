import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    x = np.asarray(x)
    gamma = np.asarray(gamma)
    beta = np.asarray(beta)

    if x.ndim == 2:
        reduce_axes = (0,)
        keep_shape = (1, -1)
    elif x.ndim == 4:
        reduce_axes = (0, 2, 3)
        keep_shape = (1, -1, 1, 1)
    else:
        raise ValueError
    mean = np.mean(x, axis=reduce_axes, keepdims=True)
    variance = np.mean((x - mean)**2, axis=reduce_axes, keepdims=True)
    norm_x = (x - mean)/np.sqrt(variance + eps)
    gamma = gamma.reshape(keep_shape)
    beta = beta.reshape(keep_shape)
    y = gamma * norm_x + beta
    return y