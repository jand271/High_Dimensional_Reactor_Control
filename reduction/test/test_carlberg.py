import numpy as np
from sklearn.decomposition import TruncatedSVD
from reduction.carlberg import Carlberg
from reduction.test.utils import assert_left_singular_vectors_equal
from unittest import TestCase


class TestWeightedPODModelReduction(TestCase):
    def test_SVD_replication_small(self):
        X = np.array([[3, 2, 2], [2, 3, -2]])
        U, S, Z = np.linalg.svd(X)

        r = Carlberg(X, 2, np.eye(2))

        assert_left_singular_vectors_equal(r.V, U, tol=1e-5)

    def test_SVD_replication_large(self):
        nx = int(1e3)

        X = np.random.random((nx, nx))

        U, S, Z = np.linalg.svd(X)

        r = Carlberg(X, nx, np.eye(nx))

        assert_left_singular_vectors_equal(r.V, U, tol=1e-2)

    def test_SVD_truncation_small(self):
        X = np.array([[3, 2, 2], [2, 3, -2]])
        r = 1

        truncated_svd = TruncatedSVD(n_components=r)
        US = truncated_svd.fit_transform(X)
        U = US / truncated_svd.singular_values_

        r = Carlberg(X, r, np.eye(2))

        assert_left_singular_vectors_equal(r.V, U, tol=1e-5)

    def test_SVD_truncation_large(self):
        nx = int(1e3)
        X = np.arange(0, nx * nx).reshape((nx, nx))
        r = 2

        truncated_svd = TruncatedSVD(n_components=r)
        US = truncated_svd.fit_transform(X)
        U = US / truncated_svd.singular_values_

        r = Carlberg(X, r, np.eye(nx))

        assert_left_singular_vectors_equal(r.V, U, tol=1e-2)

    def test_algorithm1_and_algorithm2(self):
        nx = 10
        r = 2

        V = np.random.random(size=(nx, nx))
        D = np.diag(np.arange(nx))
        X = V @ D @ np.linalg.inv(V)

        # X = np.array([[3, 2, 2, 5, -1], [2, 3, -2, 6, -1]])

        C = np.arange(2 * nx).reshape((2, nx)) + 5
        Theta = C.T @ C

        V1 = Carlberg.algorithm_1(r, X, Theta)
        V2 = Carlberg.algorithm_2(r, X, Theta)

        from reduction.demo.utils import orthogonal_error

        orthogonal_error(X, C, V1, V1)
        orthogonal_error(X, C, V2, V2)

        assert np.linalg.norm(np.abs(np.divide(V1, V2)) - np.ones(V1.shape)) < 1e-10
