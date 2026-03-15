def discount_returns(rewards, gamma):
    """
    Compute the discounted return at every timestep.
    """
    G = [0 for i in range(len(rewards))]
    n = len(rewards)
    G[n - 1] = rewards[n - 1]
    for i in range(1, n):
        G[n - i - 1] = rewards[n - i - 1] + gamma * G[n - i]
    return G