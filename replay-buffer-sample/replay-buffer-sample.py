import numpy as np
def replay_buffer_sample(buffer, batch_size, seed):
    """
    Sample a batch of transitions from the replay buffer.
    """
    rng = np.random.RandomState(seed)
    replay_index = rng.choice(len(buffer), size=batch_size, replace=False)
    return [buffer[i] for i in replay_index]