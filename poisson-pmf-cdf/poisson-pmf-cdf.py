import numpy as np
import math
def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    # Write code here
    mass = np.exp(-lam) * lam**k / math.factorial(k)
    cdf = 0
    for i in range(k + 1):
        cdf += (np.exp(-lam) * lam**i)/math.factorial(i)
    return mass, cdf