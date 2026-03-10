def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    set_A = set(set_a)
    set_B = set(set_b)
    intersect = set_A & set_B
    union = set_A | set_B
    if not union:
        return 0.0
    return len(intersect) / len(union)