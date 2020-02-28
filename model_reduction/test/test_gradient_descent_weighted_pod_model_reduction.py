import numpy as np
from sklearn.decomposition import TruncatedSVD
from model_reduction.gradient_descent_weighted_pod_model_reduction import GradientDescentWeightedPODModelReduction
from model_reduction.test.utils import assert_left_singular_vectors_equal
from unittest import TestCase


class TestGradientDescentWeightedPODModelReduction(TestCase):
    def test_SVD_replication_small(self):
        X = np.array([[3, 2, 2], [2, 3, -2]])
        U, S, Z = np.linalg.svd(X)

        r = GradientDescentWeightedPODModelReduction(X, 2,  np.eye(2))

        assert_left_singular_vectors_equal(r.V, U, tol=1e-5)

    def test_SVD_replication_large(self):
        nx = int(1e3)

        X = np.random.random((nx, nx))

        U, S, Z = np.linalg.svd(X)

        r = GradientDescentWeightedPODModelReduction(X, nx, np.eye(nx))

        assert_left_singular_vectors_equal(r.V, U, tol=1e-2)

    def test_SVD_truncation_small(self):
        X = np.array([[3, 2, 2], [2, 3, -2]])
        r = 1

        truncated_svd = TruncatedSVD(n_components=r)
        US = truncated_svd.fit_transform(X)
        U = US / truncated_svd.singular_values_

        r = GradientDescentWeightedPODModelReduction(X, r, np.eye(2))

        assert_left_singular_vectors_equal(r.V, U, tol=1e-5)

    def test_SVD_truncation_large(self):
        nx = int(1e3)
        X = np.arange(0, nx * nx).reshape((nx, nx))
        r = 2

        truncated_svd = TruncatedSVD(n_components=r)
        US = truncated_svd.fit_transform(X)
        U = US / truncated_svd.singular_values_

        r = GradientDescentWeightedPODModelReduction(X, r, np.eye(nx))

        assert_left_singular_vectors_equal(r.V, U, tol=1e-2)
