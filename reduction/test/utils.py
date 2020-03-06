import numpy as np


def assert_left_singular_vectors_equal(A, B, tol=1e-10):
    """ asserts that the two input left singular bases are equivalent """

    assert A.shape == B.shape, 'input SVD Us must have same shape'

    r = A.shape[1]

    deviation = np.linalg.norm(np.abs(A.T @ B) - np.eye(r))

    assert deviation < tol, "Asserted deviation < {0:.2e}, but deviation is {1:.2e}".format(tol, deviation)
