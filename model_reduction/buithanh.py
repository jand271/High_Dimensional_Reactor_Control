import numpy as np
import matlab.engine
from os.path import dirname
from model_reduction.model_reduction import GalerkinModelReduction


class BuiThanh(GalerkinModelReduction):
    def __init__(self, M, K, f, U, m, C=None, beta=0):
        """
        Constructor for model reduction strategy by Bui-Thanh
        :param M: Mass Matrix
        :param K: Stiffness Matrix
        :param f: Affine Term
        :param U: Snapshots
        :param m: Number of DOFs for ROM
        :param C: state to output matrix
        :param beta: regularization parameter
        """

        self.nx = M.shape[0]
        self.ns = U.shape[1]

        if C is None:
            C = np.eye(self.nx)

        """ Check Constructor Inputs """
        assert type(M) is np.ndarray, 'M must be a numpy array'
        assert type(K) is np.ndarray, 'f must be a numpy array'
        assert type(f) is np.ndarray, 'f must be a numpy array'
        assert type(U) is np.ndarray, 'U must be a numpy array'
        assert isinstance(m, int), 'requested basis rank m must be int > 0'
        assert type(C) is np.ndarray, 'C must be a numpy array'
        assert isinstance(beta, int) or isinstance(beta, float), 'regularization beta must be int or float'

        assert M.shape == (self.nx, self.nx), 'M must have shape (nx, nx)'
        assert K.shape == (self.nx, self.nx), 'K must have shape (nx, nx)'
        assert f.shape == (self.nx,), 'f must have shape (nx,)'
        assert C.shape[1] == self.nx, 'C must have shape nx columns'
        assert U.shape[0] == self.nx, 'U must have nx rows'

        self.M = M
        self.K = K
        self.f = f
        self.U = U
        self.m = m
        self.C = C
        self.H = C.T @ C
        self.beta = beta

        self.problem = None
        self.Phi = None
        self.Gamma = None
        self.alpha = None
        self.J = None
        self.constraints = None

        self.V = None

        super().__init__()

    def compute_reduction_basis(self):
        current_file_directory = dirname(__file__)
        yalmip_directory = '/Users/jasonanderson/Non-Cloud Documents/YALMIP-master'

        eng = matlab.engine.start_matlab()

        eng.eval("addpath('" + current_file_directory + "')")
        eng.eval("addpath(genpath('" + yalmip_directory + "'))")

        eng.workspace['M'] = matlab.double(self.M.tolist())
        eng.workspace['K'] = matlab.double(self.K.tolist())
        eng.workspace['f'] = matlab.double(self.f.reshape((self.nx, 1)).tolist())
        eng.workspace['U'] = matlab.double(self.U.tolist())
        eng.workspace['m'] = matlab.double([self.m])
        eng.workspace['C'] = matlab.double(self.C.tolist())
        eng.workspace['beta'] = matlab.double([self.beta])

        V = np.array(eng.eval('buithanh(M,K,f,U,m,C,beta)'))
        eng.quit()

        self.V = self.compute_closest_orthonormal_matrix(V)
