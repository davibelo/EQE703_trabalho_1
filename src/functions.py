import numpy as np
import pandas as pd
from IPython.display import display
import time

def execution_time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter_ns()
        result = func(*args, **kwargs)
        end_time = time.perf_counter_ns()
        execution_time = (end_time - start_time) / 1000  # converting nanoseconds to microseconds
        print(f"Execution Time for {func.__name__}: {execution_time} microseconds")
        return result
    return wrapper

def display_as_dataframe(array):
    """
    Display a NumPy array as a pandas DataFrame.

    Parameters:
    array (numpy.ndarray): The NumPy array to be displayed.
    """
    # Create a DataFrame from the NumPy array
    df = pd.DataFrame(array)

    # Display the DataFrame
    display(df)

@execution_time_decorator
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

def set_zero_if_below_tolerance(array, tolerance=1e-5):
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


def classify_diagonal(matrix):
    """
    Classifies the diagonal elements of a 2D NumPy array according to the following rules:
    - PD: all elements are positive
    - PSD: all elements are zero or positive
    - ND: all elements are negative
    - NSD: all elements are zero or negative
    - INDEF: array has positive, negative and zeros
    
    Args:
    matrix (np.ndarray): A 2D NumPy array.

    Returns:
    str: The classification of the diagonal elements.
    """

    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("Input must be a 2D NumPy array")

    diagonal = np.diag(matrix)

    has_positive = np.any(diagonal > 0)
    has_negative = np.any(diagonal < 0)
    has_zero = np.any(diagonal == 0)

    if has_positive and not has_negative and not has_zero:
        return "PD"
    elif has_positive and not has_negative and has_zero:
        return "PSD"
    elif not has_positive and has_negative and not has_zero:
        return "ND"
    elif not has_positive and has_negative and has_zero:
        return "NSD"
    else:
        return "INDEF"

@execution_time_decorator
def determinant_with_pivoting(matrix, tolerance=1E-6):
    """
    Calculate the determinant of a square matrix using the pivoting method,
    considering a tolerance for treating small numbers as zero.

    Parameters:
    matrix (numpy.ndarray): A square matrix.
    tolerance (float): The tolerance level for treating numbers as zero.

    Returns:
    float or None: The determinant of the matrix. Returns None if the matrix is not square.
    2D numpy array: The triangular matrix produced by the method

    Raises:
    ValueError: If the input matrix is not square or not a 2D numpy array.
    """

    # Check if the input is a 2D numpy array
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("Input must be a 2D numpy array.")

    # Check if the matrix is square
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError("Input matrix must be square.")

    # Grant elements as float
    matrix = matrix.astype(float)

    # Initialize determinant as 1
    det = 1

    # Iterating over each column
    for j in range(cols):
        pivot = matrix[j, j]
        # Consider pivot as zero if below the tolerance
        if abs(pivot) < tolerance:
            pivot = 0

        if pivot == 0:
            # Find a suitable row below with a non-zero pivot to swap with
            for k in range(j + 1, rows):
                if abs(matrix[k, j]) > tolerance:
                    matrix[[j, k]] = matrix[[k, j]]  # Swap rows
                    det *= -1  # Changing rows changes the sign of determinant
                    pivot = matrix[j, j]
                    break
            else:
                # If no suitable row found, determinant is zero
                return 0, matrix

        # Make elements below the pivot zero
        for i in range(j + 1, rows):
            factor = -(matrix[i, j] / pivot)
            matrix[i, j:] += factor * matrix[j, j:]

            # Set small values to zero based on tolerance
            matrix[i, j:][abs(matrix[i, j:]) < tolerance] = 0

        # Multiply the diagonal elements to get determinant
        det *= pivot

    return det, matrix

@execution_time_decorator
def determinant_with_numpy(matrix):
    return np.linalg.det(matrix)

def determinant_with_pivoting_norm(matrix, tolerance=1E-6):
    """
    Calculate the determinant of a square matrix using the pivoting method,
    and normalize pivots to 1 when possible, considering a tolerance for treating small numbers as zero.

    Parameters:
    matrix (numpy.ndarray): A square matrix.
    tolerance (float): The tolerance level for treating numbers as zero.

    Returns:
    float or None: The determinant of the matrix. Returns None if the matrix is not square.
    2D numpy array: The triangular matrix produced by the method

    Raises:
    ValueError: If the input matrix is not square or not a 2D numpy array.
    """

    # Check if the input is a 2D numpy array
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("Input must be a 2D numpy array.")

    # Check if the matrix is square
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError("Input matrix must be square.")

    # Grant elements as float
    matrix = matrix.astype(float)

    # Initialize determinant as 1
    det = 1

    # Iterating over each column
    for j in range(cols):
        # Select the pivot as the largest absolute value for numerical stability
        pivot_index = np.argmax(abs(matrix[j:, j])) + j
        pivot = matrix[pivot_index, j]

        # Swap the row with the highest pivot to the current row
        if pivot_index != j:
            matrix[[j, pivot_index]] = matrix[[pivot_index, j]]
            det *= -1  # Changing rows changes the sign of the determinant

        # Consider pivot as zero if below the tolerance
        if abs(pivot) < tolerance:
            return 0, matrix

        # Normalize the pivot row (divide the row by the pivot element)
        matrix[j] = matrix[j] / pivot
        det *= pivot  # Update determinant
        
        # Make elements below the pivot zero
        for i in range(j + 1, rows):
            factor = matrix[i, j]  # already zero due to normalization
            matrix[i] -= factor * matrix[j]

            # Set small values to zero based on tolerance
            matrix[i][abs(matrix[i]) < tolerance] = 0

    return det, matrix