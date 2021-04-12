import numpy as np
from reduction.model_reduction import ModelReduction


class PODModelReduction(ModelReduction):
    def __init__(self, X, r):
        """
        Computes vanilla POD model reduction basis of rank r with input snapshot matrix X
        :param X: snapshot matrix
        :param r: desired rank
        """

        """ Check constructor inputs """
        assert type(X) is np.ndarray, "snapshot matrix X must be a numpy array"
        assert isinstance(r, int), "desired rank r must be an int"

        self.X = X
        self.r = r

        self.nx = X.shape[0]
        self.ns = X.shape[1]

        super().__init__()

    def compute_reduction_basis(self):
        """ Computes reduction basis from the left singular vectors of snapshot matrix """
        self.V = self.compute_svd_left_singular_vectors(self.X, self.r)
        self.W = self.V

    def __str__(self):
        return "pod_rank_" + str(self.r)
