import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.array(y)
    uniques, counts = np.unique(y, return_counts=True)
    entropy = 0
    total_counts = len(y)
    proba = counts / total_counts
    return np.sum(-proba * np.log2(proba))