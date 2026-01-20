import numpy as np

def td_value_update(V, s, r, s_next, alpha, gamma):
    """
    Returns: updated value function V_new
    """
    # Write code here
    V = np.array(V, dtype=float)
    V_new = V.copy()
    td_error = r + gamma * V[s_next] - V[s]
    V_new[s] = V[s] + alpha * td_error
    return V_new