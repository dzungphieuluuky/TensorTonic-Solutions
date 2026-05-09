def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    # Write code here
    import numpy as np
    X = np.asarray(X)
    W = np.asarray(W)
    b = np.asarray(b)
    result = X @ W + b
    return result.tolist()