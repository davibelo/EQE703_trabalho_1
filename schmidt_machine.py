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
    for k in range(1, n):
        # Start with Wk
        Pk = W[:, k]

        # Subtract the projections of Wk onto all previous Pi (i < k)
        for i in range(k):
            Pi = P[:, i]
            Wk = W[:, k]
            alpha = -(Pi.T @ Wk) / (Pi.T @ Pi)
            Pk += alpha * Pi

        # Set the calculated Pi into the ith column of P
        P[:, k] = Pk

    for i in range(P.shape[1]):  # Iterate over columns
        vector = P[:, i]
        norm = np.linalg.norm(vector)
        if norm != 0:
            P[:, i] = vector / norm

    return P

W = np.array([[12, -51,   4],
              [ 6, 167, -68],
              [-4,  24, -41]])

P_correct = np.array([[ 6/7, -69/175, -58/175],
                      [ 3/7, 158/175,   6/175],
                      [-2/7,    6/35,  -33/35]])

P = schmidt_machine(W)

print("W = ")
print(W)
print("\nP = ")
print(P)
print("\nP_correct:")
print(P_correct)

# Compare P with P_correct
if np.allclose(P, P_correct):
    print("\nP is equal to P_correct")
else:
    print("\nP is not equal to P_correct")

D = P.T @ (W @ P)
print('D = ')
print(D)
