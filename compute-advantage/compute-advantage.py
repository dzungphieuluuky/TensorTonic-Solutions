import numpy as np

def compute_advantage(states, rewards, V, gamma):
    """
    Returns: A (NumPy array of advantages)
    """
    G = 0
    num_timesteps = len(states)
    adv = np.zeros(num_timesteps)
    for t in range(num_timesteps - 1, -1, -1):
        G = rewards[t] + gamma * G
        adv[t] = G - V[states[t]]
    return adv