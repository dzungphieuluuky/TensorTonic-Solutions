import numpy as np
def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    n = len(actual_tokens)
    pred = np.zeros(n)
    for i in range(n):
        actual_token = actual_tokens[i]
        pred[i] = prob_distributions[i][actual_token]
    H = -np.mean(np.log(pred))
    return np.exp(H)