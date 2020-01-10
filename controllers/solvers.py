import numpy as np
import cvxpy
from abc import ABC, abstractmethod


class Solver(ABC):

    def __init__(self, nx, nu, N, x0, xbar, Q=None, R=None, umax=None):
        """
        CFTOCSolver Constructor: Initializes cvxpy problem with the following params
        :param nx number of x states
        :param nu number of input variables
        :param x0: initial state
        :param xbar: preferred end state
        :param N: number of time steps per cftoc
        """

        """ Check Constructor Inputs """
        assert type(nx) is int, 'nx must be an int'
        assert type(nu) is int, 'nu must be an int'
        assert type(N) is int, 'N must be an int'
        assert type(x0) is np.ndarray, 'x0 must be a numpy array'
        assert type(xbar) is np.ndarray, 'xbar must be a numpy array'
        assert x0.ndim == 1, 'x0 must be flat'
        assert xbar.ndim == 1, 'xbar must be flat'
        assert len(x0) == nx, 'length of x0 must be nx'
        assert len(xbar) == nx, 'length of xbar must be nx'

        if Q is None:
            Q = np.eye(nx)
        else:
            assert type(Q) is np.ndarray, 'Q must be a numpy array'
            assert Q.shape == (nx, nx), 'Q must have shape (nx, nx)'

        if R is None:
            R = np.eye(nu)
        else:
            assert type(R) is np.ndarray, 'Q must be a numpy array'
            assert R.shape == (nu, nu), 'R must have shape (nu, nu)'

        # initialize cvxpy problem
        self.X = cvxpy.Variable((nx, N + 1))
        self.U = cvxpy.Variable((nu, N))
        self.x0 = cvxpy.Parameter(nx)
        self.xbar = cvxpy.Parameter(nx)
        self.problem = None

        # initialize Parameters
        self.x0.value = x0
        self.xbar.value = xbar

        # constraints
        self.constraints = []
        self.constraints.append(self.X[:, 0] == self.x0)

        # input constraints
        if umax is not None:
            self.umax = cvxpy.Parameter()
            self.umax.value = umax
            for t in range(N):
                self.constraints.append(cvxpy.norm(self.U[:, t], 'inf') <= self.umax)

        # cost function initialization
        self.cost = 0

        # state cost
        for t in range(1, N + 1):
            self.cost += cvxpy.quad_form(self.X[:, t] - self.xbar, Q)

        # input cost
        for t in range(N):
            self.cost += cvxpy.quad_form(self.U[:, t], R)

    def set_xbar(self, xbar):
        """ update new xbar value"""
        self.xbar.value = xbar

    def calculate_optimal_actuation(self, x0):
        """
        Updates x0 and xbar parameters then solves
        :param x0: updated x0
        :return: optimal actuation
        """
        self.x0.value = x0
        return self.solve_optimal_actuation()

    def solve_optimal_actuation(self):
        """
        solves cvxpy problem and returns problem solution variable
        :return: optimal actuation value
        """
        self.problem.solve(warm_start=True)
        return self.U[:, 0].value

    def status(self):
        """ returns problem status """
        return self.problem.status


class CFTOCSolver(Solver):

    def __init__(self, A, B, f, x0, xbar, N, umax=None, Q=None, R=None):
        """
        CFTOCSolver Constructor: Initializes cvxpy problem with the following params
        :param A: state transition matrix
        :param B: control matrix
        :param f: affine term in dynamic equation
        :param x0: initial state
        :param xbar: preferred end state
        :param N: number of time steps per cftoc
        :param umax: maximum allowable control actuation
        """

        nx = x0.shape[0]
        nu = B.shape[1]

        """ Check Constructor Inputs """
        assert type(A) is np.ndarray, 'A must be a numpy array'
        assert type(B) is np.ndarray, 'B must be a numpy array'
        assert type(f) is np.ndarray, 'f must be a numpy array'
        assert A.shape[0] == nx, 'row A must = number of x0 states'
        assert A.shape[0] == A.shape[1], 'A must be square'
        assert B.shape[0] == nx, 'row B must = number of x0 states'
        assert umax is None or umax > 0, 'umax must be None or greater than 0'

        super().__init__(nx, nu, N, x0, xbar, umax=umax, Q=Q, R=R)

        for t in range(N):
            self.constraints.append(self.X[:, t + 1] == A @ self.X[:, t] + B @ self.U[:, t] + f)

        self.problem = cvxpy.Problem(cvxpy.Minimize(self.cost), self.constraints)
