import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    X = np.asarray(X)
    if X.ndim < 2 or X.shape[0] < 2:
        return None
    N, D = X.shape
    mean = np.mean(X, axis=0)
    X_center = X - mean
    return (X_center.T @ X_center) / (N - 1)