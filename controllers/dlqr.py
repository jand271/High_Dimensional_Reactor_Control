import numpy as np
from scipy.linalg import inv, block_diag, solve_discrete_are
from controllers.controller import Controller


class DLQR(Controller):
    """ Infinite-horizon Discrete LQR Controller """

    def __init__(self, A, B, horizon=10, Q=None, R=None, _child_call=False):
        """
        Solve the Infinite-horizon Discrete LQR Problem and store F matrix
        :param A: state transition matrix
        :param B: control matrix
        :param horizon: controller horizon
        :param Q: state quadratic cost
        :param R: control quadratic cost
        :param _child_call: True if constructor called from child class
        """

        """ If child called constructor, no need to check constructor inputs """
        if not _child_call:
            self._check_constructor_inputs(A, B, horizon, Q, R)

        if Q is None:
            Q = np.eye(A.shape[0])

        if R is None:
            R = np.eye(B.shape[1])

        self.A = A
        self.B = B
        self.horizon = horizon
        self.Q = Q
        self.R = R

        self.F = None

        self.compute_gain_matrix()

    def _check_constructor_inputs(self, A, B, horizon, Q, R):
        """ Checks Constructor inputs """

        assert type(A) is np.ndarray, "A must be a numpy array"
        assert type(B) is np.ndarray, "B must be a numpy array"
        assert A.shape[0] == A.shape[1], "A must be square"
        assert A.shape[0] == B.shape[0], "B must have the same number of rows as A"
        assert isinstance(horizon, int) and horizon >= 0, "horizon must be int >= 0"

        (nx, nu) = B.shape

        if Q is not None:
            assert isinstance(Q, np.ndarray)
            assert Q.shape == (nx, nx), "Q.shape must be (nx, nx)"

        if R is not None:
            assert isinstance(Q, np.ndarray)
            assert R.shape == (nu, nu), "R.shape must be (nu, nu)"

    def set_desired_state(self, state):
        print("WARNING: Attempted to set_desired_state of DLQR controller. Feature not available to DLQR")

    def update_then_calculate_optimal_actuation(self, x):
        """ compute optimal output from current state x """
        return -self.F @ x

    def compute_gain_matrix(self):
        """
        Computes F matrix from A, B, Q, and R for Infinite-horizon, discrete-time LQR Controller if self.horizon=0
        otherwise compute finite-horizon case from wikipedia <https://en.wikipedia.org/wiki/Linear–quadratic_regulator>.
        """

        if self.horizon == 0:  # infinite horizon case
            Pk = solve_discrete_are(self.A, self.B, self.Q, self.R)
        else:  # finite horizon case
            Pk = self.Q
            for i in range(self.horizon):
                Pk = (
                    self.A.T @ Pk @ self.A
                    - (self.A.T @ Pk @ self.B) @ inv(self.R + self.B.T @ Pk @ self.B) @ (self.B.T @ Pk @ self.A)
                    + self.Q
                )
        self.F = inv(self.R + self.B.T @ Pk @ self.B) @ (self.B.T @ Pk @ self.A)
        return


class AffineDLQR(DLQR):
    """ Infinite-horizon Discrete LQR Controller """

    def __init__(self, A, B, f, horizon=10, Q=None, R=None, _child_call=False):
        """
        Solve the Infinite-horizon Discrete LQR Problem and store F matrix
        :param A: state transition matrix
        :param B: control matrix
        :param f: affine term
        :param horizon: controller horizon
        :param Q: state quadratic cost
        :param R: control quadratic cost
        :param _child_call: True if constructor called from child class
        """

        """ Check Constructor Inputs"""
        if not _child_call:
            self._check_constructor_inputs(A, B, f, horizon, Q, R)

        if Q is None:
            Q = np.eye((A.shape[0]))

        A_affine = np.block([[A, f[:, np.newaxis]], [np.zeros((1, A.shape[1])), 1]])
        B_affine = np.block([[B], [np.zeros((1, B.shape[1]))]])
        Q_affine = np.block([[Q, np.zeros((Q.shape[0], 1))], [np.zeros((1, Q.shape[1])), 0]])

        super().__init__(A_affine, B_affine, Q=Q_affine, R=R, _child_call=True)

    def _check_constructor_inputs(self, A, B, f, horizon, Q, R):
        """ Checks Constructor inputs """

        assert type(f) is np.ndarray, "f must be a numpy array"
        assert f.ndim == 1, "f must be flat"
        assert len(f) == A.shape[0], "f must have same number of rows as A"
        super()._check_constructor_inputs(A, B, horizon, Q, R)
        return

    def set_desired_state(self, state):
        print("WARNING: Attempted to set_desired_state of DLQR controller. Feature not available to DLQR")

    def update_then_calculate_optimal_actuation(self, x):
        """ compute optimal output from current state x """
        return -self.F @ np.hstack((x, 1))


class TrackingAffineDLQR(AffineDLQR):
    """ Infinite-horizon Discrete LQR Controller """

    def __init__(self, A, B, f, horizon=10, Q=None, R=None, xbar=None, _child_call=False):
        """
        Solve the Infinite-horizon Discrete LQR Problem and store F matrix
        :param A: state transition matrix
        :param B: control matrix
        :param f: affine term
        :param horizon: controller horizon
        :param Q: state quadratic cost
        :param R: control quadratic cost
        :param xbar: desired state for tracking
        :param _child_call: True if constructor called from child class
        """

        """ Check Constructor Inputs"""
        if xbar is None:
            xbar = np.zeros((A.shape[0],))

        if Q is None:
            Q = np.eye((A.shape[0]))

        if not _child_call:
            self._check_constructor_inputs(A, B, f, horizon, Q, R, xbar)

        self.f = f
        self.xbar = xbar

        (nx, nu) = B.shape
        A_aug = np.block([[A, np.zeros((nx, nx))], [np.eye(nx), np.eye(nx)]])
        B_aug = np.block([[B], [np.zeros((nx, nu))]])
        f_aug = np.hstack((f, -xbar))
        Q_aug = block_diag(np.zeros((nx, nx)), Q)

        super().__init__(A_aug, B_aug, f_aug, horizon=horizon, Q=Q_aug, R=R, _child_call=True)

    def _check_constructor_inputs(self, A, B, f, horizon, Q, R, xbar):
        """ Checks Constructor inputs """

        assert type(xbar) is np.ndarray, "xbar must be a numpy array"
        assert xbar.ndim == 1, "xbar must be flat"
        assert len(xbar) == A.shape[0], "xbar must have same number of rows as A"

        super()._check_constructor_inputs(A, B, f, horizon, Q, R)

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

    def __init__(self, A, B, horizon=10, Q=None, R=None, xbar=None, _child_call=False):
        """
        Solve the Infinite-horizon Discrete LQR Problem and store F matrix
        :param A: state transition matrix
        :param B: control matrix
        :param horizon: controller horizon
        :param Q: state quadratic cost
        :param R: control quadratic cost
        :param xbar: desired state for tracking
        :param _child_call: True if constructor called from child class
        """
        super().__init__(A, B, np.zeros((A.shape[0],)), horizon=horizon, Q=Q, R=R, xbar=xbar, _child_call=False)
