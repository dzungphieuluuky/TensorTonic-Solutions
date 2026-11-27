import numpy as np

def knn_distance(X_train: list, X_test: list, k: int) -> np.ndarray:
    """
    Returns a NumPy array with shape (n_test, k).
    """
    # Write code here
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    if X_train.ndim == 1:
        X_train = X_train[:, None]
        X_test = X_test[:, None]
    dist = np.sum((X_test[:, None, :] - X_train[None, :, :])**2, axis=-1)
    dist = np.argsort(dist, axis=-1)
    num_test, num_train = dist.shape
    padded = np.concatenate([dist, np.ones((num_test, k)) * -1], axis=-1)
    return padded[:, :k]
        