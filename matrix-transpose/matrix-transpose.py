import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    (n,m) = np.array(A).shape
    new = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            new[i][j] = A[j][i]
    return new
