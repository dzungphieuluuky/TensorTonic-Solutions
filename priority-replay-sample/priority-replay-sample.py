import numpy as np
def priority_replay_sample(priorities, alpha, beta):
    """
    Compute sampling probabilities and importance sampling weights for PER.
    """
    priorities = np.asarray(priorities)
    N = priorities.shape[0]
    power = priorities ** alpha
    Z = np.sum(power)
    proba = power / Z
    raw_weights = 1 / (N * proba)**beta
    normalized_weights = raw_weights / np.max(raw_weights)
    return [proba.tolist(), normalized_weights.tolist()]