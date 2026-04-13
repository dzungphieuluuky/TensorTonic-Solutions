def expected_calibration_error(y_true, y_pred, n_bins):
    """
    Compute Expected Calibration Error.
    """
    # Write code here
    width = 1/n_bins
    total = len(y_pred)
    bins = [[] for i in range(n_bins)]
    for i in range(total):
        index = int(y_pred[i]//width)
        if index == n_bins:
            bins[-1].append((y_pred[i], y_true[i]))
        else:
            bins[index].append((y_pred[i], y_true[i]))
    ece = 0
    for bin in bins:
        if len(bin) == 0:
            continue
        coeff = len(bin) / total
        acc = 0
        conf = 0
        for (pred, label) in bin:
            acc += label
            conf += pred 
        acc /= len(bin)
        conf /= len(bin)
        ece += coeff * (acc - conf) * (2 * (acc > conf) - 1)
    return ece
    