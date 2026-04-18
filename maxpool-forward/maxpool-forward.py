def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    # Write code here
        # Write code here
    import numpy as np
    X = np.array(X)
    h, w = X.shape
    out_h = (h - pool_size)//stride + 1
    out_w = (w - pool_size)//stride + 1
    result = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            result[i, j] = np.max(X[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size])
    return result.tolist()