import numpy as np

def knn_distance(X_train, X_test, k):
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # If 1D (like [1, 3, 5]), reshape to (3, 1). If 2D, leave it.
    if X_train.ndim == 1:
        X_train = X_train[:, np.newaxis]
    if X_test.ndim == 1:
        X_test = X_test[:, np.newaxis]

    n_train = X_train.shape[0]

    # Broadcasting: (M, 1, D) - (N, D) -> (M, N, D)
    diff = X_test[:, np.newaxis, :] - X_train
    
    # Calculate Euclidean distance
    dist = np.linalg.norm(diff, axis=2)

    # Get sorted indices (M, N)
    indices = np.argsort(dist, axis=1)

    # Handle k > n_train by padding with -1
    if k > n_train:
        padding = np.full((indices.shape[0], k - n_train), -1)
        indices = np.hstack([indices, padding])

    return indices[:, :k]