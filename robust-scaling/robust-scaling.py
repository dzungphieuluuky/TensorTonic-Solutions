import numpy as np

def get_median(sorted_vals):
    n = len(sorted_vals)
    if n % 2 == 1:
        return sorted_vals[n // 2]
    else:
        return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2

def robust_scaling(values):
    values = np.asarray(values)
    if values.size == 0:
        return []

    # Special case: single value
    if values.size == 1:
        return [0.0]

    # Sort for robust statistics
    sorted_vals = np.sort(values)

    # Compute median and split into lower/upper halves
    n = len(sorted_vals)
    if n % 2 == 1:          # odd number of values
        median = sorted_vals[n // 2]
        lower = sorted_vals[:n // 2]
        upper = sorted_vals[n // 2 + 1:]
    else:                   # even number of values
        median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        lower = sorted_vals[:n // 2]
        upper = sorted_vals[n // 2:]

    # Compute quartiles (median of lower and upper halves)
    q1 = get_median(lower)
    q3 = get_median(upper)
    iqr = q3 - q1

    # Apply scaling
    if iqr == 0:
        scaled = values - median
    else:
        scaled = (values - median) / iqr

    return scaled.tolist()