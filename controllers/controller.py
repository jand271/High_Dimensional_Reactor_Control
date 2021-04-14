import numpy as np
from scipy.linalg import inv, block_diag, solve_discrete_are
from abc import ABC, abstractmethod


class Controller(object):
    @abstractmethod
    def set_desired_state(self, state):
        """ set desired state to track """
        pass

    @abstractmethod
    def update_then_calculate_optimal_actuation(self, state):
        """
        calculates controller output
        :param state: current state
        :return: controller command
        """
        pass
