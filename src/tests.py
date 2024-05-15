import unittest
import numpy as np
from functions import *

class TestFunctions(unittest.TestCase):

    def test_schmidt_machine(self):
        W = np.array([[12, -51, 4],
                      [6, 167, -68],
                      [-4, 24, -41]])

        P_correct = np.array([[6/7, -69/175, -58/175],
                              [3/7, 158/175, 6/175],
                              [-2/7, 6/35, -33/35]])

        P = schmidt_machine(W)

        self.assertTrue(np.allclose(P, P_correct), "P is not equal to P_correct")

    def test_set_zero_if_below_tolerance(self):
        array = np.array([[0.5, 1.2, 0.03],
                          [2.3, 0.04, 0.8],
                          [0.1, 0.02, 1.5]])

        tolerance = 0.05
        modified_array_custom = set_zero_if_below_tolerance(array, tolerance)

        expected_array = np.array([[0.5, 1.2, 0.0],
                                   [2.3, 0.0, 0.8],
                                   [0.1, 0.0, 1.5]])

        np.testing.assert_array_equal(modified_array_custom, expected_array,
                                      "The modified array is not as expected")
    def test_determinant_calculation(self):
        matrix = np.array([[ 1, 2,  0, 1],
                           [ 2, 4, -1, 0],
                           [ 3, 2,  0, 0],
                           [-1, 0,  1, 1]])
        expected_det = 6
        calculated_det = determinant_with_pivoting(matrix)
        self.assertEqual(calculated_det, expected_det)

if __name__ == '__main__':

    unittest.main()
