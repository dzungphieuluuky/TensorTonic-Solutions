def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    count = {}
    words = set()
    for sentence in sentences:
        for word in sentence:
            if word in count.keys():
                count[word] += 1
            else:
                count[word] = 1
    return count