import numpy as np

def policy_gradient_loss(log_probs, rewards, gamma):
    """
    REINFORCE loss with mean-return baseline.

    Args:
        log_probs: list/array of log probabilities (length T)
        rewards: list/array of rewards (length T)
        gamma: discount factor

    Returns:
        scalar loss value
    """
    # Input validation
    if len(log_probs) != len(rewards):
        raise ValueError("log_probs and rewards must have same length")
    if len(rewards) == 0:
        return 0.0

    log_probs = np.asarray(log_probs)
    rewards = np.asarray(rewards)

    # Compute discounted returns (backward)
    returns = np.zeros_like(rewards, dtype=float)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        returns[t] = running

    # Advantage with baseline = mean(returns)
    advantage = returns - np.mean(returns)

    # Loss = - mean(advantage * log_prob)
    loss = -np.mean(advantage * log_probs)
    return loss.item()