import numpy as np

def schmidt_machine(W):
    """Applies the Schmidt process to matrix A to orthogonalize its columns."""

    # Number of vectors (columns)
    n = A.shape[1]

    # Initialize the orthogonal matrix with zeros
    P = np.zeros_like(W)

    # Make P0 = W0
    P0 = W[:, 0]
    P[:, 0] = P0

    W1 = W[:, 1]
    alfa10 = -(P0.T @ W1) / (P0.T @ P0)
    P1 = W1 + alfa10 * P0
    P[:, 1] = P1

    W2 = W[:, 2]
    alfa20 = -(P0.T @ W2) / (P0.T @ P0)
    alfa21 = -(P1.T @ W2) / (P1.T @ P1)
    P2 = W2 + alfa20 * P0 + alfa21 * P1
    P[:, 2] = P2
    print(P)

    return P

W = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(W)

schmidt_machine(W)
# print(B)
