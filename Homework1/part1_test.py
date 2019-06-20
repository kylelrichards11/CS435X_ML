from homework1_template import *
import numpy as np

A = np.array([[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12],[13, 14, 15, 16]])
B = np.array([[1, 2], [6, 7]])


def test_problem1():
    ans = np.array([[2, 4, 6, 8], [10, 12, 14, 16], [18, 20, 22, 24], [26, 28, 30, 32]])
    assert np.array_equal(problem1(A, A), ans)
