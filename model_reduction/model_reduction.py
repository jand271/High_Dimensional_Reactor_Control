import numpy as np
from numpy.linalg import svd
from sklearn.decomposition import TruncatedSVD
from abc import abstractmethod
from scipy.io import savemat


class ModelReduction(object):
    def __init__(self, W=None, V=None):
        """
        Model reduction class
        :param W: Left subspace
        :param V: Right subspace
        """

        """ Check Constructor Inputs """
        assert W is None or W is np.ndarray, "W must be a np.ndarray"
        assert V is None or V is np.ndarray, "V must be a np.ndarray"

        self.W = W
        self.V = V

        if V is None or W is None:
            assert V is None and W is None, "Why is just V or just W None?"
            self.compute_reduction_bases()

    @abstractmethod
    def compute_reduction_bases(self):
        pass


class GalerkinModelReduction(ModelReduction):
    def __init__(self, V=None):
        """
        Galerkin Model reduction constructor
        :param V: reduction basis x^{full} = V x^{reduced}
        """
        super().__init__(W=V, V=V)

    def compute_reduction_bases(self):
        self.compute_reduction_basis()
        self.W = self.V

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
