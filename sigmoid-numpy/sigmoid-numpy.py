import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    
    y[pos_mask] = 1/ (1+np.exp(-x[pos_mask]))
    
    y[neg_mask] = np.exp(x[neg_mask])/(1+np.exp(x[neg_mask]))
    return y
        