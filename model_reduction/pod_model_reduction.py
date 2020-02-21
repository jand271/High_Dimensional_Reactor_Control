import numpy as np
from model_reduction.model_reduction import ModelReduction


class PODModelReduction(ModelReduction):

    def __init__(self, A, B, C, r, X):
        """
        Computes vanilla POD model reduction basis of rank r on the input dynamic system with input snapshot matrix
        :param A: state transition matrix
        :param B: input to state matrix
        :param C: output matrix
        :param r: desired rank
        :param X: snapshot matrix
        """

        """ Check constructor inputs """
        assert isinstance(r, int), "desired rank r must be an int"
        assert type(X) is np.ndarray, 'snapshot matrix X must be a numpy array'
        assert A.shape[0] == X.shape[0], 'snapshot matrix X must have the same number of rows as A'

        self.r = r
        self.X = X
        self.ns = X.shape[1]

        super().__init__(A, B, C)

    def compute_reduction_basis(self):
        """ Computes reduction basis from the left singular vectors of snapshot matrix """
        self.V = self.compute_svd_left_singular_vectors(self.X, self.r)
