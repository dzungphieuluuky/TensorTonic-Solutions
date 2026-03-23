import numpy as np

def euclidean_distance(x, y):
    if x.ndim == 1:
        return np.sum((x - y) ** 2)
    else:
        return np.sum((x - y) ** 2, axis=1)
def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    """
    anchor = np.asarray(anchor)
    positive = np.asarray(positive)
    negative = np.asarray(negative)
    d_pos = euclidean_distance(anchor, positive)
    d_neg = euclidean_distance(anchor, negative)
    L = np.maximum(0, d_pos - d_neg + margin)
    return np.mean(L)
    
    