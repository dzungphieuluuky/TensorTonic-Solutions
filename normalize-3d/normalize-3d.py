import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    v = np.array(v)
    dims = v.ndim
    norm = np.linalg.norm(v, axis=dims - 1, keepdims=True)
    return np.where(norm == 0, 0, v / norm)