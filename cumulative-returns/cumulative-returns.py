def cumulative_returns(returns):
    """
    Compute the cumulative return at each time step.
    """
    # Write code here
    results = []
    current = 1
    for item in returns:
        current *= (1 + item)
        results.append(current - 1)
    return results