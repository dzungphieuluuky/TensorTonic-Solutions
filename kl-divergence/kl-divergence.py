import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    # Write code here
    p = np.array(p)
    q = np.array(q)

    ready_to_log_p = np.where(p == 0, 1, p)
    return np.sum(p * np.log(ready_to_log_p/(q + eps)))