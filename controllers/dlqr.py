import numpy as np
from scipy.linalg import inv, block_diag, solve_discrete_are
from controllers.controller import Controller


class DLQR(Controller):
    """ Infinite-horizon Discrete LQR Controller """

    def __init__(self, A, B, Q=None, R=None, _child_call=False):
        """
        Solve the Infinite-horizon Discrete LQR Problem and store F matrix
        :param A: state transition matrix
        :param B: control matrix
        :param Q: state quadratic cost
        :param R: control quadratic cost
        :param tolerance: error change iteration tolerance when solving Riccati Equation
        :param max_iter: maximum number of iterations through dynamic Riccati Equation
        :param _child_call: True if constructor called from child class
        """

        """ If child called constructor, no need to check constructor inputs """
        if not _child_call:
            self._check_constructor_inputs(A, B, Q=Q, R=R)

        if Q is None:
            Q = np.eye(A.shape[0])

        if R is None:
            R = np.eye(B.shape[1])

        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

        self.F = None

        self.compute_gain_matrix()

    def _check_constructor_inputs(self, A, B, Q=None, R=None):
        """ Checks Constructor inputs """

        assert type(A) is np.ndarray, 'A must be a numpy array'
        assert type(B) is np.ndarray, 'B must be a numpy array'
        assert A.shape[0] == A.shape[1], 'A must be square'
        assert A.shape[0] == B.shape[0], 'B must have the same number of rows as A'

        (nx, nu) = B.shape

        if Q is not None:
            assert isinstance(Q, np.ndarray)
            assert Q.shape == (nx, nx), 'Q.shape must be (nx, nx)'

        if R is not None:
            assert isinstance(Q, np.ndarray)
            assert R.shape == (nu, nu), 'R.shape must be (nu, nu)'

    def set_desired_state(self, state):
        print("WARNING: Attempted to set_desired_state of DLQR controller. Feature not available to DLQR")

    def update_then_calculate_optimal_actuation(self, x):
        """ compute optimal output from current state x """
        return -self.F @ x

    def compute_gain_matrix(self):
        """ Computes F matrix from A, B, Q, and R for Infinite-horizon, discrete-time LQR Controller """
        P = solve_discrete_are(self.A, self.B, self.Q, self.R)
        self.F = inv(self.R + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A)
        return


class AffineDLQR(DLQR):
    """ Infinite-horizon Discrete LQR Controller """

    def __init__(self, A, B, f, Q=None, R=None, tolerance=1e-5, min_iter=10, max_iter=15, _child_call=False):
        """
        Solve the Infinite-horizon Discrete LQR Problem and store F matrix
        :param A: state transition matrix
        :param B: control matrix
        :param f: affine term
        :param Q: state quadratic cost
        :param R: control quadratic cost
        :param tolerance: error change iteration tolerance when solving Riccatti Equation
        :param min_iter: minimum number of iterations through dunamic Riccatti Equation
        :param max_iter: maximum number of iterations through dunamic Riccatti Equation
        :param _child_call: True if constructor called from child class
        """

        """ Check Constructor Inputs"""
        if not _child_call:
            self._check_constructor_inputs(A, B, f, Q=Q, R=R, tolerance=tolerance, min_iter=min_iter, max_iter=max_iter)

        if Q is None:
            Q = np.eye((A.shape[0]))

        self.tolerance = tolerance
        self.min_iter = min_iter
        self.max_iter = max_iter

        A_affine = np.block([[A, f[:, np.newaxis]], [np.zeros((1, A.shape[1])), 1]])
        B_affine = np.block([[B], [np.zeros((1, B.shape[1]))]])
        Q_affine = np.block([[Q, np.zeros((Q.shape[0], 1))], [np.zeros((1, Q.shape[1])), 0]])

        super().__init__(A_affine, B_affine, Q=Q_affine, R=R, _child_call=True)

    def _check_constructor_inputs(self, A, B, f, Q=None, R=None, tolerance=1e-5, min_iter=10, max_iter=15):
        """ Checks Constructor inputs """

        assert type(f) is np.ndarray, 'f must be a numpy array'
        assert f.ndim == 1, 'f must be flat'
        assert len(f) == A.shape[0], 'f must have same number of rows as A'
        assert isinstance(tolerance, float), "tolerance must be a float"
        assert isinstance(min_iter, int) and max_iter > 0, "min_iter must be int > 0"
        assert isinstance(max_iter, int) and max_iter > 0, "max_iter must be int > 0"
        super()._check_constructor_inputs(A, B, Q=Q, R=R)

    def compute_gain_matrix(self):
        """
        Computes F matrix from A, B, Q, and R for Infinite-horizon, discrete-time LQR Controller according to
        wikipedia <https://en.wikipedia.org/wiki/Linear–quadratic_regulator>.

        SciPy has trouble solving DARE with affine dynamics
        """

        Pk = np.random.rand(*self.A.shape)
        Fk = inv(self.R + self.B.T @ Pk @ self.B) @ (self.B.T @ Pk @ self.A)
        previous_error = np.inf
        for i in range(self.max_iter):
            Pk1 = self.A.T @ Pk @ self.A - \
                  (self.A.T @ Pk @ self.B) @ inv(self.R + self.B.T @ Pk @ self.B) @ (self.B.T @ Pk @ self.A) + self.Q
            if i >= self.min_iter:
                Fk1 = inv(self.R + self.B.T @ Pk1 @ self.B) @ (self.B.T @ Pk1 @ self.A)
                error = np.linalg.norm(Fk1 - Fk)
            if i >= self.min_iter and abs(previous_error - error) < self.tolerance:
                self.F = inv(self.R + self.B.T @ Pk1 @ self.B) @ (self.B.T @ Pk1 @ self.A)
                return
            else:
                Pk = Pk1
                if i >= self.min_iter:
                    #print("Riccati Equation iterative solution error change: {0:f}".format(abs(previous_error - error)))
                    Fk = Fk1
                    previous_error = error

        # raise RuntimeError("Unable to converge while solving Riccati Equation.")
        print("WARNING: Unable to converge while solving Riccati Equation.")
        self.F = inv(self.R + self.B.T @ Pk1 @ self.B) @ (self.B.T @ Pk1 @ self.A)
        return

    def set_desired_state(self, state):
        print("WARNING: Attempted to set_desired_state of DLQR controller. Feature not available to DLQR")

    def update_then_calculate_optimal_actuation(self, x):
        """ compute optimal output from current state x """
        return -self.F @ np.hstack((x, 1))


class TrackingAffineDLQR(AffineDLQR):
    """ Infinite-horizon Discrete LQR Controller """

    def __init__(self, A, B, f, Q=None, R=None, xbar=None, tolerance=1e-5, min_iter=10, max_iter=15, _child_call=False):
        """
        Solve the Infinite-horizon Discrete LQR Problem and store F matrix
        :param A: state transition matrix
        :param B: control matrix
        :param f: affine term
        :param Q: state quadratic cost
        :param R: control quadratic cost
        :param tolerance: error change iteration tolerance when solving Riccatti Equation
        :param max_iter: maximum number of iterations through dunamic Riccatti Equation
        :param _child_call: True if constructor called from child class
        """

        """ Check Constructor Inputs"""
        if xbar is None:
            xbar = np.zeros((A.shape[0],))

        if Q is None:
            Q = np.eye((A.shape[0]))

        if not _child_call:
            self._check_constructor_inputs(A, B, f, Q=Q, R=R, xbar=xbar, tolerance=tolerance, min_iter=min_iter,
                                           max_iter=max_iter)

        self.f = f
        self.xbar = xbar

        (nx, nu) = B.shape
        A_aug = np.block([[A, np.zeros((nx, nx))], [np.eye(nx), np.eye(nx)]])
        B_aug = np.block([[B], [np.zeros((nx, nu))]])
        f_aug = np.hstack((f, -xbar))
        Q_aug = block_diag(np.zeros((nx, nx)), Q)

        super().__init__(A_aug, B_aug, f_aug, Q=Q_aug, R=R, tolerance=tolerance, min_iter=min_iter, max_iter=max_iter,
                         _child_call=True)

    def _check_constructor_inputs(self, A, B, f, Q=None, R=None, xbar=None, tolerance=1e-5, min_iter=10, max_iter=15):
        """ Checks Constructor inputs """

        assert type(xbar) is np.ndarray, 'xbar must be a numpy array'
        assert xbar.ndim == 1, 'xbar must be flat'
        assert len(xbar) == A.shape[0], 'xbar must have same number of rows as A'

        super()._check_constructor_inputs(A, B, f, Q=None, R=None, tolerance=tolerance, min_iter=min_iter,
                                          max_iter=max_iter)

    def set_desired_state(self, xbar):
        """ set new xbar and recompute gain matrix """

        assert xbar.shape == self.xbar.shape
        self.xbar = xbar

        self.A[:, -1] = np.hstack((self.f, -self.xbar, 1))
        self.compute_gain_matrix()

    def update_then_calculate_optimal_actuation(self, x):
        """ compute optimal output from current state x """

        return -self.F @ np.hstack((x, x - self.xbar, 1))


class TrackingDLQR(TrackingAffineDLQR):
    """ Infinite-horizon Discrete LQR Controller """

    def __init__(self, A, B, Q=None, R=None, xbar=None, tolerance=1e-5, min_iter=10, max_iter=15, _child_call=False):
        """
        Solve the Infinite-horizon Discrete LQR Problem and store F matrix
        :param A: state transition matrix
        :param B: control matrix
        :param Q: state quadratic cost
        :param R: control quadratic cost
        :param tolerance: error change iteration tolerance when solving Riccatti Equation
        :param max_iter: maximum number of iterations through dunamic Riccatti Equation
        :param _child_call: True if constructor called from child class
        """
        super().__init__(A, B, np.zeros((A.shape[0],)), Q=Q, R=R, xbar=xbar, tolerance=tolerance,
                         min_iter=min_iter, max_iter=max_iter, _child_call=False)
