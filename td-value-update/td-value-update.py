import numpy as np

def td_value_update(V, s, r, s_next, alpha, gamma):
    """
    Returns: updated value function V_new
    """
    target = r + gamma * V[s_next]
    delta = target - V[s]
    V[s] += alpha * delta
    return V