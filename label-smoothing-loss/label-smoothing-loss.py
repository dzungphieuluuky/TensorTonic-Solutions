def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.
    """
    # Write code here
    import numpy as np
    k = len(predictions)
    predictions = np.array(predictions)
    not_target = epsilon/k
    smooth_target = np.full((k), not_target)
    smooth_target[target] = (1 - epsilon) + epsilon/k
    return -np.sum(smooth_target * np.log(predictions))