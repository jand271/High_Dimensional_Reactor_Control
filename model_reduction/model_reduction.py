import numpy as np
from numpy.linalg import svd
from sklearn.decomposition import TruncatedSVD
from abc import abstractmethod
from scipy.io import savemat


class ModelReduction(object):
    def __init__(self, V=None):
        """
        Abstract class model reduction
        :param V: reduction basis x^{full} = V x^{reduced}
        """

        self.V = None

        if V is None:
            self.compute_reduction_basis()
        else:
            assert V is np.ndarray, "V must be a np.ndarray"
            self.V = V

        self.compute_reduction_basis()

    def save_reduction_basis(self, directory):
        """ Saves reduction basis to directory """
        assert isinstance(directory, str), 'reduction basis save directory input must be a string'
        assert directory[-4:] == '.mat', "path must end in '.mat'"
        savemat(directory, {'V': self.V})

    @abstractmethod
    def compute_reduction_basis(self):
        pass

    @staticmethod
    def compute_truncated_svd(X, r):
        """ :return matrix of r left singular vectors of X """
        # Truncated SVD throws an error if requested rank is the same as the data matrix column; hence, call full svd
        # if needed
        if r == X.shape[1]:
            U, S, ZT = svd(X)
        elif r < X.shape[1]:
            truncated_svd = TruncatedSVD(n_components=r)
            US = truncated_svd.fit_transform(X)
            U = US / truncated_svd.singular_values_
            S = truncated_svd.singular_values_
            ZT = truncated_svd.components_
        else:
            raise ValueError("Requested SVD rank r cannot be larger than the columns of the input matrix X")
        return U, S, ZT

    @staticmethod
    def compute_svd_left_singular_vectors(X, r):
        """ :return matrix of r left singular vectors of X """
        # Truncated SVD throws an error if requested rank is the same as the data matrix column; hence, call full svd
        # if needed
        if r == X.shape[1]:
            U, S, ZT = svd(X)
        elif r < X.shape[1]:
            truncated_svd = TruncatedSVD(n_components=r)
            US = truncated_svd.fit_transform(X)
            U = US / truncated_svd.singular_values_
        else:
            raise ValueError("Requested SVD rank r cannot be larger than the columns of the input matrix X")
        return U

    @staticmethod
    def compute_closest_orthonormal_matrix(X):
        """ :return: returns closest orthonormal matrix via Orthogonal Procrustes Problem """
        U, S, ZT = np.linalg.svd(X)
        I = np.eye(*X.shape)
        return U @ I @ ZT
