import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    v = np.array(v)
    if v.ndim == 1:
        norm = np.linalg.norm(v)
    elif v.ndim == 2:
        norm = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(norm == 0, 0, v / norm)