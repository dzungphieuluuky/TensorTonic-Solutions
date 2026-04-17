import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    """
    Compute confusion matrix with optional normalization.
    """
    # Write code here
    n = len(y_true)
    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)
    if num_classes is None:
        num_classes = np.max(y_true) + 1
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.zeros((num_classes, num_classes))
    confusion_matrix = np.zeros((num_classes, num_classes))
    flatten_index = y_true * num_classes + y_pred
    counts_flatten_index = np.bincount(flatten_index, minlength=num_classes**2)
    if len(confusion_matrix) == 0:
        return None
    confusion_matrix = counts_flatten_index.reshape((num_classes, num_classes))
    if normalize == "none":
        return confusion_matrix
    elif normalize == "true":
        sum_row = np.sum(confusion_matrix, axis=1, keepdims=True)
        sum_row = np.where(sum_row == 0, 1, sum_row)
        return confusion_matrix/sum_row
    elif normalize == "pred":
        sum_column = np.sum(confusion_matrix, axis=0, keepdims=True)
        sum_column = np.where(sum_column == 0, 1, sum_column)
        return confusion_matrix/sum_column
    elif normalize == "all":
        sum_all = np.sum(confusion_matrix, keepdims=True)
        sum_all = np.where(sum_all == 0, 1, sum_all)
        return confusion_matrix/sum_all
    return confusion_matrix
    
    