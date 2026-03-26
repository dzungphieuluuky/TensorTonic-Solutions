def is_all_equal(values):
    pivot = values[0]
    for value in values:
        if value != pivot:
            return False
    return True
    
def binning(values, num_bins):
    """
    Assign each value to an equal-width bin.
    """
    n = len(values)
    if is_all_equal(values):
        return [0 for i in range(n)]
    min_val = min(values)
    max_val = max(values)
    w = (max_val - min_val) / num_bins
    res = []
    for i in range(n):
        bin = min(((values[i] - min_val) // w), num_bins - 1)
        res.append(bin)
    return res
    