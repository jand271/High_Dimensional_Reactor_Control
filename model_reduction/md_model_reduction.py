import numpy as np
from model_reduction.model_reduction import GalerkinModelReduction


class MDModelReduction(GalerkinModelReduction):

    def __init__(self, A_full, r):
        """
        Computes vanilla POD model reduction basis of rank r on the input dynamic system with input snapshot matrix
        :param A_full: state transition matrix for full model
        :param r: desired rank
        """

        """ Check constructor inputs """
        assert type(A_full) is np.ndarray or A_full is None, 'A must be a numpy array'
        assert isinstance(r, int), "desired rank r must be an int"
        assert A_full.shape[0] == A_full.shape[1], 'A must be square'

        self.A_full = A_full
        self.r = r

        super().__init__()

    def compute_reduction_basis(self):
        """ Computes reduction basis from the left singular vectors of state transition matrix """
        self.V = self.compute_svd_left_singular_vectors(self.A_full, self.r)
