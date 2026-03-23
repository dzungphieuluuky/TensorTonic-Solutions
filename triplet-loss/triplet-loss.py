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

    L = np.maximum(0, euclidean_distance(anchor, positive) - euclidean_distance(anchor, negative) + margin)
    return np.mean(L)
    
    