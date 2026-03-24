import numpy as np
def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    recommended = np.array(recommended)
    relevant = np.array(relevant)
    k_recommend = recommended[:k]
    k_recommend = set(k_recommend)
    relevant = set(relevant)
    intersect = k_recommend & relevant
    return [len(intersect)/k, len(intersect)/len(relevant)]