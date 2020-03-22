import numpy as np
from unittest import TestCase
from reduction.test.utils import assert_left_singular_vectors_equal
from reduction.model_reduction import ModelReduction


class TestModelReduction(TestCase):
    def test_compute_truncated_svd(self):
        X = np.random.random(size=(10, 5))

        U, S, VT = np.linalg.svd(X)
        U_t1 = U[:, :3]
        S_t1 = np.diag(S[:3])
        VT_t1 = VT[:3, :]

        U_t2, S_t2, VT_t2 = ModelReduction.compute_truncated_svd(X, 3)

        assert np.linalg.norm(U_t1 @ S_t1 @ VT_t1 - U_t2 @ np.diag(S_t2) @ VT_t2) < 1e-10

    def test_compute_svd_left_singular_vectors(self):
        X = np.random.random(size=(10, 5))

        U, S, VT = np.linalg.svd(X)

        U_truncated = ModelReduction.compute_svd_left_singular_vectors(X, 3)

        assert_left_singular_vectors_equal(U[:, :3], U_truncated)
