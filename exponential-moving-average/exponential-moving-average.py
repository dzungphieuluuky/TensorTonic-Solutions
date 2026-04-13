def exponential_moving_average(values, alpha):
    """
    Compute the exponential moving average of the given values.
    """
    # Write code here
    results = []
    results.append(values[0])
    ema = results[0]
    for i in range(1, len(values)):
        ema = alpha * values[i] + (1 - alpha) * ema
        results.append(ema)
    return results