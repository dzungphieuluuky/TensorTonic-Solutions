import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    counts = {}
    for word in vocab:
        counts[word] = 0
    for token in tokens:
        if token not in vocab:
            continue 
        counts[token] += 1
    res = []
    if len(vocab) == 0:
        return np.array(res, dtype=int)
    for word in vocab:
        res.append(counts[word])
    return np.array(res, dtype=int)