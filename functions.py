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


def are_columns_orthogonal(matrix, tolerance=1e-10):
    """
    Checks if the columns of the given matrix are orthogonal to each other within a specified tolerance.

    Parameters:
    matrix (numpy.ndarray): A 2D NumPy array where each column is a vector.
    tolerance (float): The tolerance within which the dot product is considered zero.

    Returns:
    bool: True if all columns are orthogonal to each other within the specified tolerance, False otherwise.
    """
    num_columns = matrix.shape[1]

    for i in range(num_columns):
        for j in range(i + 1, num_columns):
            dot_product = np.dot(matrix[:, i], matrix[:, j])
            if abs(dot_product) > tolerance:
                print(
                    f"Columns {i} and {j} are not orthogonal. Dot product: {dot_product}"
                )
                return False
            else:
                print(
                    f"Columns {i} and {j} are orthogonal within tolerance. Dot product: {dot_product}"
                )

    return True

def set_zero_if_below_tolerance(array, tolerance=1E-10):
    """
    Sets elements of a 2D NumPy array to zero if they are below a specified tolerance.

    Parameters:
    array (np.ndarray): Input 2D NumPy array.
    tolerance (float, optional): The tolerance threshold. Default is 1E-10.

    Returns:
    np.ndarray: The modified array with elements below tolerance set to zero.
    """
    # Ensure the input is a NumPy array
    array = np.array(array)

    # Apply the condition using boolean indexing
    array[array < tolerance] = 0

    return array
