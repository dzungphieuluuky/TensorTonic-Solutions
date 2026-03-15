import numpy as np
def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    x1, x2 = np.asarray(x1), np.asarray(x2)
    norm_1 = np.sqrt(np.sum(x1 ** 2))
    norm_2 = np.sqrt(np.sum(x2 ** 2))
    cosine = np.dot(x1, x2) / (norm_1 * norm_2)
    if label == 1:
        return 1 - cosine
    else:
        return max(0, cosine - margin)