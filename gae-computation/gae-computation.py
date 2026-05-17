def gae(rewards, values, gamma, lam):
    """
    Compute Generalized Advantage Estimation.
    """
    # Write code here
    deltas = []
    for i in range(len(values) - 1):
        delta = rewards[i] + gamma * values[i + 1] - values[i]
        deltas.append(delta)
    advantages = [0 for i in range(len(deltas))]
    for i  in range(len(deltas) - 1, -1, -1):
        if i == len(deltas) - 1:
            advantages[i] = deltas[i]
        else:
            advantages[i] = deltas[i] + gamma * lam * advantages[i + 1]
    return advantages