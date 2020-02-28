import numpy as np
from abc import ABC
from model_reduction.pod_model_reduction import PODModelReduction


class WeightedPODModelReduction(ABC, PODModelReduction):

    def __init__(self, X, r, C_full):
        """
        Constructor for weighted POD model reduction
        :param X: snapshot matrix
        :param r: desired rank
        :param C_full: state to output matrix of full order model

        """

        """ Check constructor inputs """
        assert type(C_full) is np.ndarray

        self.C_full = C_full

        super().__init__(X, r)
