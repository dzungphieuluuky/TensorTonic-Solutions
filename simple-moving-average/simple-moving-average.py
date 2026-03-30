def simple_moving_average(values, window_size):
    """
    Compute the simple moving average of the given values.
    """
    n = len(values)
    res = [0 for i in range(n - window_size + 1)]
    for i in range(n - window_size + 1):
        temp = 0
        for j in range(i, i + window_size):
            temp += values[j]
        temp /= window_size
        res[i] = temp
    return res