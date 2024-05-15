import numpy as np
from functions import *

# function: schmidt machine
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

# function: set zero if below tolerance
array = np.array([[0.5, 1.2, 0.03],
                  [2.3, 0.04, 0.8],
                  [0.1, 0.02, 1.5]])

tolerance = 0.05
modified_array_custom = set_zero_if_below_tolerance(array, tolerance)
print('\nModified array with custom tolerance:')
print(modified_array_custom)
