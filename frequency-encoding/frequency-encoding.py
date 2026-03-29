def frequency_encoding(values):
    """
    Replace each value with its frequency proportion.
    """
    n = len(values)
    counter = {}
    for val in values:
        if val not in counter.keys():
            counter[val] = 1
        else:
            counter[val] += 1
    res = []
    for val in values:
        res.append(counter[val]/n)
    return res