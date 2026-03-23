import numpy as np
def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = y_pred.shape[0]
    if np.sum(y_pred == y_true) == n:
        return 1.0
    true_pos = np.sum(y_pred == y_true)
    false_pos_neg = y_true.shape[0] - true_pos
    return 2 * true_pos / (2 * true_pos + false_pos_neg * 2)
    