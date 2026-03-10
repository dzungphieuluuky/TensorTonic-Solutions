import numpy as np

def vector_norm_3d(v):
    """
    Compute the Euclidean norm of 3D vector(s).
    """
    v = np.array(v)
    if v.shape == (3, ):
        results = v[0]**2 + v[1]**2 + v[2]**2
        results = np.sqrt(results)
        return results
    results = v[:, 0]**2 + v[:, 1]**2 + v[:, 2]**2
    results = np.sqrt(results)
    return results