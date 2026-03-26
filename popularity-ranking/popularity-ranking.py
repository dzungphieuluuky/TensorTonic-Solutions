def popularity_ranking(items, min_votes, global_mean):
    """
    Compute the Bayesian weighted rating for each item.
    """
    result = []
    for item in items:
        rating = item[0]
        vote_count = item[1]
        new_rating = rating * (vote_count / (vote_count + min_votes)) + global_mean * (min_votes) / (vote_count + min_votes)
        result.append(new_rating)
    return result