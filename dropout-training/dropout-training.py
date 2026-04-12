import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.array(x)
    if rng is None:
        pattern = np.random.random(x.shape)
    else:
        pattern = rng.random(x.shape)
    pattern = np.where(pattern <= p, 0, 1/(1-p))
    output = x * pattern
    return output, pattern