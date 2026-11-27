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
    num_train = X_train.shape[0]
    num_test = X_test.shape[0]
    if num_train == 0 or k <= 0:
        return np.full((num_test, max(0, k)), -1, dtype=int)

    # 1. Compute squared Euclidean distance without 3D broadcasting:
    # ||x_test - x_train||^2 = ||x_test||^2 + ||x_train||^2 - 2 * (X_test @ X_train.T)
    # Fast MSE calculation, without materializaing the whole 3d matrix
    test_sq = np.sum(X_test**2, axis=1, keepdims=True)
    train_sq = np.sum(X_train**2, axis=1)
    dist_sq = test_sq + train_sq - 2 * (X_test @ X_train.T)

    # 2. If k >= num_train, just sort everything and pad with integer -1
    if k >= num_train:
        indices = np.argsort(dist_sq, axis=-1)
        pad_width = k - num_train
        if pad_width > 0:
            padding = np.full((num_test, pad_width), -1, dtype=int)
            return np.concatenate([indices, padding], axis=-1)
        return indices

    # 3. Use argpartition to find the k smallest distances in O(N) time
    part_idx = np.argpartition(dist_sq, kth=k - 1, axis=-1)[:, :k]
    
    # 4. Sort only the k partitioned neighbors so they are returned in exact order
    part_dist = np.take_along_axis(dist_sq, part_idx, axis=-1)
    sorted_order = np.argsort(part_dist, axis=-1)
    
    return np.take_along_axis(part_idx, sorted_order, axis=-1)

        