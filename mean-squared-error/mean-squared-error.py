import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    if y_pred.shape != y_true.shape:
        return None
    res = y_pred - y_true
    res = res ** 2
    return np.sum(res) / y_pred.shape[0]
