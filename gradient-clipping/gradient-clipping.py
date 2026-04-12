import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    # Write code here
    g = np.array(g)
    grad_norm = np.linalg.norm(g)
    if max_norm <= 0:
        return g
    if grad_norm > max_norm:
        new_g = g * max_norm/grad_norm
    else:
        new_g = g
    return new_g