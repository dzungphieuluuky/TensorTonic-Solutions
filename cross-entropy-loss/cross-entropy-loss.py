import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_samples = len(y_true)
    n_labels = np.max(y_true) + 1
    onehot = np.zeros((n_samples, n_labels))
    onehot[range(n_samples), y_true] = 1
    return np.mean(-np.sum(onehot * np.log(y_pred), axis=1, keepdims=True))
    