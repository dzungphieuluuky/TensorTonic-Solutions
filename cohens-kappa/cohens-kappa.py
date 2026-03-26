import numpy as np

def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    rater1 = np.array(rater1)
    rater2 = np.array(rater2)
    num_agree = np.sum(rater1 == rater2)
    n = rater1.shape[0]
    p0 = num_agree / n
    uniques1, counts1 = np.unique(rater1, return_counts=True)
    uniques2, counts2 = np.unique(rater2, return_counts=True)
    pe = 0
    for i in range(len(uniques1)):
        for j in range(len(uniques2)):
            if uniques1[i] != uniques2[j]:
                continue
            pe += (counts1[i] * counts2[j])
    pe /= n**2
    if pe == 1:
        return 1
    return (p0 - pe) / (1 - pe)