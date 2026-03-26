def percent_change(series):
    """
    Compute the fractional change between consecutive values.
    """
    res = []
    for i in range(len(series) - 1):
        if series[i] == 0:
            res.append(0)
            continue
        res.append((series[i + 1] - series[i]) / series[i])
    return res