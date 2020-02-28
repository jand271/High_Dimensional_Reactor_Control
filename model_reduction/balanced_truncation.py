import numpy as np
import matlab.engine
from model_reduction.model_reduction import ModelReduction


class BalancedTruncation(ModelReduction):
    def __init__(self, A, B, C, r):
        """
        Balanced Truncation constructor as wrapper to matlab implementation by Joseph Lorenzetti
        :param A: state transition Matrix
        :param B: control to state matrix
        :param C: state to output matrix
        :param r: reduced model rank
        """

        self.nx = A.shape[0]
        self.nu = B.shape[1]
        self.ny = C.shape[0]

        """ Check Constructor Inputs"""
        assert isinstance(A, np.ndarray), "A must be numpy.ndarray"
        assert isinstance(B, np.ndarray), "B must be numpy.ndarray"
        assert isinstance(C, np.ndarray), "C must be numpy.ndarray"
        assert isinstance(r, int) and r > 0, "r must be int > 0"

        assert A.shape == (self.nx, self.nx), "A must be shape (nx, nx)"
        assert B.shape == (self.nx, self.nu), "B must be shape (nx, nu)"
        assert C.shape == (self.ny, self.nx), "C must be shape (ny, nx)"

        self.A = A
        self.B = B
        self.C = C
        self.r = r

        super().__init__()

    def compute_reduction_bases(self):
        source_file_directory = \
            '/Users/jasonanderson/OneDrive - Leland Stanford Junior University/coursework/JA_AA290/joe_source'

        eng = matlab.engine.start_matlab()

        eng.eval("addpath('" + source_file_directory + "')")

        eng.workspace['A'] = matlab.double(self.A.tolist())
        eng.workspace['B'] = matlab.double(self.B.tolist())
        eng.workspace['C'] = matlab.double(self.C.tolist())
        eng.workspace['r'] = matlab.double([self.r])

        eng.eval("[~, ~, ~, W, V, ~] = balancedTruncationDiscrete(A, B, C, r);", nargout=0)

        self.W = np.array(eng.workspace['W'])
        self.V = np.array(eng.workspace['V'])

        eng.quit()

    def __str__(self):
        return 'balanced_truncation_rank_' + str(self.r)
