import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):
    """
    Returns: (boot_means, lower, upper)
    """
    # Write code here
    x = np.asarray(x)
    n = len(x)

    if rng is None:
        rng = np.random

    indices = rng.choice(n, size=(n_bootstrap, n), replace=True)
    values = x[indices]
    boot_means = np.mean(values, axis=1)
    alpha = 1 - ci
    lower = np.quantile(boot_means, alpha / 2)
    upper = np.quantile(boot_means, 1 - alpha/2)
    return boot_means, lower, upper
    
