import numpy as np

def chi2_independence(C):
    """
    Compute chi-square test statistic and expected frequencies.
    """
    # Write code here
    C = np.asarray(C)
    rows = np.sum(C, axis=1)
    cols = np.sum(C, axis=0)
    total = np.sum(C)

    outer = np.outer(rows, cols)
    E = outer / total
    chi = np.sum((C - E)**2 / E)
    return float(chi), E