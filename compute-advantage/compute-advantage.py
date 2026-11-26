import numpy as np

def compute_advantage(states: list, rewards: list, V: list, gamma: float) -> np.ndarray:
    """
    Returns the advantages as a NumPy array.
    """
    # Write code here
    G = 0
    num_timesteps = len(states)
    advantage = np.zeros(num_timesteps)
    for t in range(num_timesteps - 1, -1, -1):
        G = rewards[t] + gamma * G
        advantage[t] = G - V[states[t]]
    return np.round(advantage, 4)
