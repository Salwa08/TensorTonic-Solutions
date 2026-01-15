import numpy as np

def q_learning_update(Q, s, a, r, s_next, alpha, gamma):
    """
    Returns: updated Q-table Q_new
    """
    # Write code here
    Q = np.array(Q, dtype=float)
    Q_new = Q.copy()
    
    if s_next < 0:
        target = r
    else:
        target = r + gamma * np.max(Q[s_next])
        
    Q_new[s, a] = Q[s, a] + alpha * (target - Q[s, a])
    return Q_new