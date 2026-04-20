def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """
    # Write code here
    import numpy as np
    predictions = np.array(predictions)
    targets = np.array(targets)
    new_predictions = predictions * targets + (1 - predictions) * (1 - targets)
    focal = np.mean(-alpha * (1 - new_predictions)**gamma * np.log(new_predictions))
    return focal
    