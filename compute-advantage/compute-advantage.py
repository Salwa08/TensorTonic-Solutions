import numpy as np

def compute_advantage(states, rewards, V, gamma):
    """
    Returns: A (NumPy array of advantages)
    """
    # Write code here
    A = []
    T = len(states)
    for t in range(T):
        cumulative_reward = 0.0
        for k in range(t, T):
            cumulative_reward += rewards[k] * gamma**(k-t) 
  
        A.append(cumulative_reward - V[states[t]])

    return np.array(A, dtype=float)

