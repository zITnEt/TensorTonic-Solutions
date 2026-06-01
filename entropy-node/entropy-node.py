import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    
    _, cnts = np.unique(y, return_counts=True)
    p = cnts/cnts.sum()

    return (np.log2(p)*(-p)).sum().astype(float)