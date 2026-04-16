import numpy as np

def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def information_gain(y, split_mask):
    """
    Compute Information Gain of a binary split on labels y.
    Use the _entropy() helper above.
    """
    # Write code here
    array_y = np.array(y)
    split_mask = np.array(split_mask)
    left = (split_mask == True)
    right = (split_mask == False)

    n_l = np.sum(left)
    n_r = np.sum(right)
    if n_l == 0 or n_r == 0:
        return 0
    n = len(y)

    return _entropy(array_y) - (n_l*_entropy(array_y[left]) + n_r*_entropy(array_y[right]))/n
