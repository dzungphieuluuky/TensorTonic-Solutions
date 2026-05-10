def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    # Write code here
    L = np.sqrt(6 / (fan_in + fan_out))
    new_w = np.array(W) * 2 * L - L
    return new_w