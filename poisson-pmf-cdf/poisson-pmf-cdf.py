import math
import numpy
def poisson_pmf_cdf(lam: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    # Write code here
    mass = np.exp(-lam) * lam**k / math.factorial(k)
    cdf = 0
    for i in range(k + 1):
        cdf += (np.exp(-lam) * lam**i)/math.factorial(i)
    return {
        "pmf": float(mass),
        "cdf": float(cdf)
    }
