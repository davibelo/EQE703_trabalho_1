import numpy as np


def schmidt_machine(W):
    """Applies the Schmidt process to matrix A to orthogonalize its columns vectors."""

    # Grant elements as float
    W = W.astype(float)

    # Number of vectors (columns)
    n = W.shape[1]

    # Initialize the orthogonal matrix with zeros
    P = np.zeros_like(W)

    # Make P0 = W0
    P0 = W[:, 0]
    P[:, 0] = P0

    # Loop to calculate P1, P2, ..., Pn-1
    for i in range(1, n):
        # Start with Wi
        Pi = W[:, i]

        # Subtract the projections of Wi onto all previous Pj (j < i)
        for j in range(i):
            Pj = P[:, j]
            alpha = -(Pj.T @ W[:, i]) / (Pj.T @ Pj)
            Pi += alpha * Pj

        # Set the calculated Pi into the ith column of P
        P[:, i] = Pi

    return P

W = np.array([[1, 1, 0],
              [0, 1, 1],
              [1, 0, 1]])

P_correct = np.array([[1,  0.5, -2/3],
                      [0,    1,  2/3],
                      [1, -0.5,  2/3]])

P = schmidt_machine(W)

print("W:")
print(W)
print("\nP:")
print(P)
print("\nP_correct:")
print(P_correct)

# Compare P with P_correct
if np.allclose(P, P_correct):
    print("\nP is equal to P_correct")
else:
    print("\nP is not equal to P_correct")
