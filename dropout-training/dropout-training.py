import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here

    x = np.array(x, dtype=float)
    r = rng.random(x.shape) if rng is not None else np.random.random(x.shape)
    pattern = ((r > p).astype(float))/(1-p)
    output = x*pattern

    return output, pattern