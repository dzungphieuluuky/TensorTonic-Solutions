import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    # Write code here
    matrix = np.array(matrix)
    if axis is not None and matrix.ndim <= axis:
        return None
    if axis is not None and matrix.shape[axis] == 1:
        return None
    if norm_type == "l2":
        euclid = np.sqrt(np.sum(matrix**2, axis=axis, keepdims=True))
        norm_matrix = np.where(euclid == 0, 0, matrix / euclid)
    elif norm_type == "l1":
        manhattan = np.sum(np.abs(matrix), axis=axis, keepdims=True)
        norm_matrix = np.where(manhattan == 0, 0, matrix / manhattan)
    elif norm_type == "max":
        maximum = np.max(matrix, axis=axis, keepdims=True)
        norm_matrix = np.where(maximum == 0, 0, matrix / maximum)
    else:
        return None
    return norm_matrix