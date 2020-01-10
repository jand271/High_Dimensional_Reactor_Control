import numpy as np
from modeling.fem_model import FEMModel
from controllers.solvers import CFTOCSolver
from controllers.model_reduction import proper_orthogonal_decomposition_model_reduction

class HeatExchangerMPCController(object):
    def __init__(self, fem_model, initial_temperature, horizon=10, max_removal_power_density=1, Q=None, R=None):
        """ Check Constructor Inputs """
        assert isinstance(fem_model, FEMModel)
        assert isinstance(initial_temperature, int) or isinstance(initial_temperature, float)
        assert isinstance(horizon, int)
        assert isinstance(max_removal_power_density, int) or isinstance(max_removal_power_density, float)

        A, B, f = fem_model.state_transition_model()
        T0 = initial_temperature * np.ones((A.shape[0],))

        phi, A, B, f, Q = proper_orthogonal_decomposition_model_reduction(A, B, f, 25, Q=Q)
        T0 = phi.T @ T0
        self.phi = phi

        self.solver = CFTOCSolver(A, B, f, T0, T0, horizon, umax=max_removal_power_density, Q=Q, R=R)

    def set_desired_state(self, state):
        """ set xbar of solver """
        self.solver.set_xbar(state)

    def update_then_calculate_optimal_actuation(self, current_position):
        """
        calculates optimal input for controller
        :param current_position: current position
        :return: optimal velocity command
        """
        return self.solver.calculate_optimal_actuation(self.phi.T @ current_position)
