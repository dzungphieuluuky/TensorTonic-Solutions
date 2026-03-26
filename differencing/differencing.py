def first_order(series):
    result = []
    for i in range(len(series) - 1):
        result.append(series[i + 1] - series[i])
    return result
    
def differencing(series, order):
    """
    Apply d-th order differencing to the time series.
    """
    current = series
    for i in range(order):
        current = first_order(current)
    return current