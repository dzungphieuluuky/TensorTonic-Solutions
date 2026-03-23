import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)
    num_seq = len(seqs)
    final_seq = np.full((num_seq, max_len), pad_value)
    for i in range(num_seq):
        this_length = min(len(seqs[i]), max_len)
        final_seq[i][:this_length] = seqs[i][:this_length]
    return final_seq
    
    