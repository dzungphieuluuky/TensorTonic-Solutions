import math

def label_smoothing_loss(predictions: list, target: int, epsilon: float) -> float:
    """
    Returns cross-entropy loss for the smoothed target distribution.
    """
    # Write code here
    import numpy as np
    k = len(predictions)
    predictions = np.array(predictions)
    not_target = epsilon/k
    smooth_target = np.full((k), not_target)
    smooth_target[target] = (1 - epsilon) + epsilon/k
    return -np.sum(smooth_target * np.log(predictions))
