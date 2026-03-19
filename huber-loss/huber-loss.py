import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    e = y_true - y_pred
    e = np.where(abs(e) <= delta, e**2/2, delta*(abs(e) - delta/2))
    return np.mean(e)