def linear_interpolation(values):
    """
    Fill missing (None) values using linear interpolation.
    """
    # Write code here
    new_values = values.copy()
    left, right = 0, 0
    n = len(new_values)
    for i in range(n):
        if new_values[i] is not None:
            left = i
        else:
            for j in range(i + 1, n):
                if new_values[j] is not None:
                    right = j
                    break 
            new_values[i] = new_values[left] + (i - left)*(new_values[right] - new_values[left])/(right - left)
            left = i
    return new_values