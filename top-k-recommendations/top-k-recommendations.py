def top_k_recommendations(scores, rated_indices, k):
    """
    Return indices of top-k unrated items by predicted score.
    """
    score_idx = []
    for i in range(len(scores)):
        score_idx.append((scores[i], i))
    score_idx.sort(key=lambda x: x[0], reverse=True)
    res = []
    ok = 0
    for (score, idx) in score_idx:
        if idx in rated_indices:
            continue
        if ok == k:
            break
        res.append(idx)
        ok += 1
    return res