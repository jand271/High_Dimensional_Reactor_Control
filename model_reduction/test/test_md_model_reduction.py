import numpy as np
from unittest import TestCase
from numpy.linalg import svd
from model_reduction.md_model_reduction import MDModelReduction
from model_reduction.test.utils import assert_left_singular_vectors_equal


class TestMDModelReduction(TestCase):
    def test_small(self):
        nx = 5
        nu = 3
        ny = 2
        r = 3

        A = np.random.random((nx, nx))
        B = np.random.random((nx, nu))
        C = np.random.random((ny, nx))

        U, S, ZT = svd(A)

        r1 = MDModelReduction(A, B, C, nx)
        r2 = MDModelReduction(A, B, C, r)

        assert_left_singular_vectors_equal(U, r1.V)
        assert_left_singular_vectors_equal(U[:, 0:r], r2.V)
