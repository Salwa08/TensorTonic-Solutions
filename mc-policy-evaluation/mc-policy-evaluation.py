import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    """
    Returns: V (NumPy array of shape (n_states,))
    """
    # Write code here
    V = np.zeros(n_states)          # Value table
    counts = np.zeros(n_states)     # Number of first-visits per state

    for episode in episodes:
        visited = set()             # Track first visits in this episode

        for t in range(len(episode)):  # Forward iteration for first-visit MC
            state, reward = episode[t]

            if state not in visited:
                visited.add(state)

                # Compute return G_t from this timestep to the end
                G = 0
                discount = 1
                for k in range(t, len(episode)):
                    G += discount * episode[k][1]
                    discount *= gamma

                # Incremental mean update
                counts[state] += 1
                V[state] += (G - V[state]) / counts[state]

    return V