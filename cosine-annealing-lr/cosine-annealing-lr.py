import numpy as np
def cosine_annealing_schedule(base_lr, min_lr, total_steps, current_step):
    """
    Compute the learning rate using cosine annealing.
    """
    if current_step == 0:
        return base_lr
    if current_step == total_steps:
        return min_lr
    return min_lr + (base_lr - min_lr)*(1 + np.cos(np.pi * current_step / total_steps)) / 2