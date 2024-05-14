import numpy as np

def schmidt_machine(W):
    """Applies the Schmidt process to matrix A to orthogonalize its columns."""

    # Grant elements as float
    W = W.astype(float)

    # Number of vectors (columns)
    n = W.shape[1]

    # Initialize the orthogonal matrix with zeros
    P = np.zeros_like(W)

    # Make P0 = W0
    P0 = W[:, 0]
    P[:, 0] = P0

    W1 = W[:, 1]
    alfa10 = -(P0.T @ W1) / (P0.T @ P0)
    print('alfa10', alfa10)
    P1 = W1 + alfa10 * P0
    P[:, 1] = P1

    W2 = W[:, 2]
    alfa20 = -(P0.T @ W2) / (P0.T @ P0)
    alfa21 = -(P1.T @ W2) / (P1.T @ P1)
    print('alfa20', alfa20)
    print('alfa21', alfa21)
    P2 = W2 + alfa20 * P0 + alfa21 * P1
    P[:, 2] = P2

    return P

W = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])

P = np.array([[0, 1, 0],
              [0, 0, 1],
              [1, 0, 0]])

print(W)

# P = schmidt_machine(W)
print(P)

print(W.T @ P)
