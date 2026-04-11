def moving_median(values, window_size):
    """
    Compute the rolling median for each window position.
    """
    # Write code here
    n = len(values)
    results = [0 for i in range(n - window_size + 1)]
    for i in range(n - window_size + 1):
        temp = sorted(values[i:i+window_size])
        if window_size & 1 == 0:
            results[i] = ((temp[window_size//2] + temp[window_size//2 - 1]) / 2)
        else:
            results[i] = (temp[window_size//2])
    return results