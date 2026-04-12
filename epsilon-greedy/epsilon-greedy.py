import numpy as np

def epsilon_greedy(q_values, epsilon, rng=None):
    """
    Returns: action index (int)
    """
    # Write code here
    q_values = np.array(q_values)
    if rng is None:
        eps = np.random.random()
    else:
        eps = rng.random()
    if eps == 1:
        return rng.integers(len(q_values))
    elif eps == 0:
        return np.argmax(q_values)
    if eps <= epsilon:
        action = rng.integers(len(q_values))
    else:
        action = np.argmax(q_values)
    return action
