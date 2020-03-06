import numpy as np
from unittest import TestCase
from numpy.linalg import svd
from reduction.md_model_reduction import MDModelReduction
from reduction.test.utils import assert_left_singular_vectors_equal


class TestMDModelReduction(TestCase):
    def test_small(self):
        nx = 5
        r = 3

        A = np.random.random((nx, nx))

        U, S, ZT = svd(A)

        r1 = MDModelReduction(A, nx)
        r2 = MDModelReduction(A, r)

        assert_left_singular_vectors_equal(U, r1.V)
        assert_left_singular_vectors_equal(U[:, 0:r], r2.V)
