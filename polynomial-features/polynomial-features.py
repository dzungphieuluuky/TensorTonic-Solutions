def polynomial_features(values, degree):
    """
    Generate polynomial features for each value up to the given degree.
    """
    n = len(values)
    res = []
    for value in values:
        temp = [value ** i for i in range(degree + 1)]
        res.append(temp)
    return res