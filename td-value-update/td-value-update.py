import numpy as np

def td_value_update(V, s, r, s_next, alpha, gamma):
    """
    Perform a single TD(0) update.
    
    Parameters:
        V (ndarray): current value function (will be modified in-place)
        s (int): current state index
        r (float): reward received
        s_next (int or None): next state index, None if terminal
        alpha (float): step size
        gamma (float): discount factor
        terminal (bool): whether s_next is terminal
    
    Returns:
        None (V is updated in-place)
    """
    target = r + gamma * V[s_next]
    delta = target - V[s]
    V[s] += alpha * delta
    return V