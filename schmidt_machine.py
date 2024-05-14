import numpy as np


def schmidt_machine(W, M=None):
    """
    Applies the Schmidt process to a matrix W to orthogonalize its column vectors
    or calculate a conjugated base of matrix M.

    Parameters:
    -----------
    W : numpy.ndarray
        The matrix whose column vectors are to be orthogonalized.
    
    M : numpy.ndarray, optional
        The matrix to be conjugated of P

    Returns:
    --------
    numpy.ndarray
        A matrix P where the column vectors are orthogonalized versions 
        of column vectors in W or a conjugated base with matrix M (if M is given).
    """

    # Grant elements as float
    W = W.astype(float)

    # Number of vectors (columns)
    n = W.shape[1]

    # Initialize the orthogonal matrix with zeros
    P = np.zeros_like(W)

    # Make P0 = W0
    P0 = W[:, 0]
    P[:, 0] = P0

    # Set M as identity matrix if not provided
    if M is None:
        M = np.eye(W.shape[0])

    # Loop to calculate P1, P2, ..., Pn-1
    for k in range(1, n):
        # Start with Wk
        Pk = W[:, k]

        # Subtract the projections of Wk onto all previous Pi (i < k)
        for i in range(k):
            Pi = P[:, i]
            Wk = W[:, k]
            alpha = -(Pi.T @ M @ Wk) / (Pi.T @ M @ Pi)
            Pk += alpha * Pi

        # Set the calculated Pi into the ith column of P
        P[:, k] = Pk

    # Normalize columns of P
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
