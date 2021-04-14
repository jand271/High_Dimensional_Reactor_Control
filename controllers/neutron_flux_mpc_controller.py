import numpy as np
from modeling.fem_model import FEMModel
from controllers.solvers import CFTOCSolver, SoftCFTOCSolver


class NeutronFluxMPCController(object):
    def __init__(self, fem_model, initial_flux, horizon=10, max_removal_neutron_source=1, Q=None, R=None):
        """ Check Constructor Inputs """
        assert isinstance(fem_model, FEMModel)
        assert isinstance(initial_flux, int) or isinstance(initial_flux, float)
        assert isinstance(horizon, int)
        assert isinstance(max_removal_neutron_source, int) or isinstance(max_removal_neutron_source, float)

        A, B, f = fem_model.state_transition_model()
        PHI0 = initial_flux * np.ones((A.shape[0],))

        # self.solver = CFTOCSolver(A, B, f, PHI0, PHI0, horizon, umax=max_removal_neutron_source, Q=Q, R=R)
        self.solver = SoftCFTOCSolver(
            A, B, f, PHI0, PHI0, horizon, umin=-max_removal_neutron_source, umax=0, penalty=1e6, Q=Q, R=R
        )

    def set_desired_state(self, state):
        """ set xbar of solver """
        self.solver.set_xbar(state)

    def update_then_calculate_optimal_actuation(self, current_state):
        """
        calculates optimal input for controller
        :param current_state: current position
        :return: optimal velocity command
        """
        return self.solver.calculate_optimal_actuation(current_state)
