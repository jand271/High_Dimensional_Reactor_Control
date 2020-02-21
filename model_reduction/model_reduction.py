import numpy as np
from numpy.linalg import svd
from sklearn.decomposition import TruncatedSVD
from abc import ABC, abstractmethod


class ModelReduction(ABC):
    def __init__(self, A, B, C):
        """
        Abstract class model reduction
        :param A: state transition matrix
        :param B: input to state matrix
        :param C: output matrix
        """

        self.nx = A.shape[0]

        """ Check Constructor Inputs """
        assert type(A) is np.ndarray, 'A must be a numpy array'
        assert type(B) is np.ndarray, 'B must be a numpy array'
        assert type(C) is np.ndarray, 'C must be a numpy array'
        assert A.shape == (self.nx, self.nx), 'A must be shape (nx,nx)'
        assert B.shape[0] == self.nx, 'B must have nx rows'
        assert C.shape[1] == self.nx, 'C must have nx columns'

        """ Full Dynamic system """
        self.A_full = A
        self.B_full = B
        self.C_full = C

        self.V = None

        """ Reducted Dynamic System """
        self.A_red = None
        self.B_red = None
        self.C_red = None

        self.compute_reduction_basis()

    @abstractmethod
    def compute_reduction_basis(self):
        pass

    def compute_reduced_model(self):
        """ computes the reduced dynamic matrix system from the reduction basis """

        if self.V is None:
            self.compute_reduction_basis()

        self.A_red = self.V.T @ self.A_full @ self.V
        self.B_red = self.V.T @ self.B_full
        self.C_red = self.V.T @ self.C_full

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
