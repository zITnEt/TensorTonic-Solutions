import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here

    if max_len is None:
        if (len(seqs)):
            max_len = 0
        else:
            return np.zeros((0, 0), dtype=int)
        
        for seq in seqs:
            max_len = max(max_len, len(seq))

        if max_len==0:
            return np.zeros((len(seqs), 0), dtype=int)
    
    for i, seq in enumerate(seqs):
        if len(seq) > max_len:
            seqs[i] = seq[:max_len].copy()
        else:
            seqs[i] = np.pad(seq, (0, max_len-len(seq)), constant_values=pad_value);
        
    return np.array(seqs)