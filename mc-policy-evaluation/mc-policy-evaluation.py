import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    """
    Estimate the state‑value function V(s) using first‑visit Monte‑Carlo returns.

    Parameters
    ----------
    episodes : list
        A list of episodes, where each episode is a list of (state, reward) tuples.
    gamma : float
        Discount factor (0 ≤ gamma ≤ 1).
    n_states : int
        Number of distinct states, assumed to be labeled 0 … n_states‑1.

    Returns
    -------
    V : np.ndarray
        NumPy array of shape (n_states,) containing the estimated value for each state.
    """
    # Container for all returns observed for each state
    returns = [[] for _ in range(n_states)]

    # Process each episode independently
    for ep in episodes:
        # Track which states have already been visited in this episode
        visited = set()

        # Pre‑compute the discounted return from each time step to the end of the episode
        # This avoids recomputing the sum for every first‑visit.
        rewards = [r for (_, r) in ep]
        # G[t] = r[t] + gamma * r[t+1] + gamma^2 * r[t+2] + …
        G = np.zeros(len(rewards))
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = rewards[t] + gamma * running
            G[t] = running

        # Iterate through the episode and record first‑visit returns
        for t, (state, _) in enumerate(ep):
            if state not in visited:
                visited.add(state)
                # Append the pre‑computed return for this time step
                returns[state].append(G[t])

    # Compute the average return for each state; if a state was never visited,
    # its value defaults to 0.0
    V = np.zeros(n_states, dtype=float)
    for s in range(n_states):
        if returns[s]:
            V[s] = np.mean(returns[s])
        else:
            V[s] = 0.0

    return V