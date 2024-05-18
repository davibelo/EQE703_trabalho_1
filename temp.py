import numpy as np
from src.functions import *

matrix = np.array([
    [0., 1., 1., 1., 1.],
    [1., 1., 3., 4., 5.],
    [1., 3., 5., 10., 15.],
    [1., 4., 10., 19., 35.],
    [1., 5., 15., 35., 69.]
])
red_matrix = row_reduced_echelon_form(matrix)
print(red_matrix)

