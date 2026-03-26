import numpy as np
def weighted_moving_average(values, weights):
    """
    Compute the weighted moving average using the given weights.
    """
    n = len(values)
    window_size = len(weights)
    num_iter = n - window_size + 1
    result = []
    total_weight = np.sum(weights)
    for i in range(num_iter):
        current = 0
        for j in range(window_size):
            current += (values[i + j] * weights[j])
        current /= total_weight
        result.append(current)
    
    return result