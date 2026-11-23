import numpy as np

def time_based_split(timestamps: list, train_end: int, gap: int, val_end: int) -> dict:
    """
    Returns a dictionary with train_idx and val_idx as sorted integer NumPy arrays.
    """
    train_idx = []
    val_idx = []

    val_start = train_end + gap

    for i, ts in enumerate(timestamps):
        if ts < train_end:
            train_idx.append(i)
        elif val_start <= ts < val_end:
            val_idx.append(i)

    return {
        "train_idx": np.array(train_idx, dtype=int),
        "val_idx": np.array(val_idx, dtype=int),
    }