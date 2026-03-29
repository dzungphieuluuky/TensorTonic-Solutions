def remove_stopwords(tokens, stopwords):
    """
    Returns: list[str] - tokens with stopwords removed (preserve order)
    """
    result = []
    for token in tokens:
        if token not in stopwords:
            result.append(token)
    return result