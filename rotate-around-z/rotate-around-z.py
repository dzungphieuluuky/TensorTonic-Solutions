import numpy as np
def rotate_around_z(points, theta):
    """
    Rotate 3D point(s) around the Z-axis by angle theta (radians).
    
    Parameters:
    - points: (N, 3) array or (3,) array
    - theta: rotation angle in radians
    
    Returns:
    - Rotated points with same shape as input
    """
    points = np.asarray(points)
    original_shape = points.shape
    if points.ndim == 1:
        points = points.reshape(1, -1)
    
    batch, dim = points.shape
    assert dim == 3, f"Expected 3D points, got dimension {dim}"
    
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s, 0],
                    [s,  c, 0],
                    [0,  0, 1]])
    
    rotated = (rot @ points.T).T
    
    return rotated.reshape(original_shape)