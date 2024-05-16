import numpy as np
from src.functions import *

A = np.array([[0.8187581, 1., 1., 1., 1.], [1., 1.8187581, 3., 4., 5.],
              [1., 3., 5.8187581, 10., 15.], [1., 4., 10., 19.8187581, 35.],
              [1., 5., 15., 35., 69.8187581]])
det, matrix = determinant_with_pivoting(A)
print(matrix)
