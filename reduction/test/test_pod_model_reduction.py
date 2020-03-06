import numpy as np
from sklearn.decomposition import TruncatedSVD
from reduction.pod_model_reduction import PODModelReduction
from reduction.test.utils import assert_left_singular_vectors_equal
from unittest import TestCase


class TestPODModelReduction(TestCase):
    def test_SVD_replication_small(self):
        X = np.array([[3, 2, 2], [2, 3, -2]])
        U, S, Z = np.linalg.svd(X)

        r = PODModelReduction(X, 2)

        assert_left_singular_vectors_equal(r.V, U, tol=1e-5)

    def test_SVD_replication_large(self):
        nx = int(1e3)

        X = np.random.random((nx, nx))

        U, S, Z = np.linalg.svd(X)

        r = PODModelReduction(X, nx)

        assert_left_singular_vectors_equal(r.V, U, tol=1e-2)

    def test_SVD_truncation_small(self):
        X = np.array([[3, 2, 2], [2, 3, -2]])
        r = 1

        truncated_svd = TruncatedSVD(n_components=r)
        US = truncated_svd.fit_transform(X)
        U = US / truncated_svd.singular_values_

        r = PODModelReduction(X, r)

        assert_left_singular_vectors_equal(r.V, U, tol=1e-5)

    def test_SVD_truncation_large(self):
        nx = int(1e3)
        X = np.arange(0, nx * nx).reshape((nx, nx))
        r = 2

        truncated_svd = TruncatedSVD(n_components=r)
        US = truncated_svd.fit_transform(X)
        U = US / truncated_svd.singular_values_

        r = PODModelReduction(X, r)

        assert_left_singular_vectors_equal(r.V, U, tol=1e-2)
