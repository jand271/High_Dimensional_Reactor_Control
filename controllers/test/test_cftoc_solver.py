import numpy as np
import unittest
from controllers.solvers import CFTOCSolver, SoftCFTOCSolver


class PositionMPCController2D:
    """ Basic 2D Position MPC Controller """

    def __init__(self, initial_position=np.zeros((3,)), horizon=10, step_time=0.1, max_speed=0.2 / 3, soft=False):
        """
        Initialized controller
        :param initial_position: initial position
        :param horizon: MPC horizon
        :param step_time: time in between MPC steps
        :param max_speed: maximum allowable speed
        """
        self._horizon = horizon
        self._step_time = step_time

        A = np.eye(3)  # state dynamics
        B = step_time * np.eye(3)  # input velocity dynamics
        f = np.zeros((3,))

        if soft:
            self.solver = SoftCFTOCSolver(
                A, B, f, initial_position, initial_position, horizon, umin=-max_speed, umax=max_speed
            )
        else:
            self.solver = CFTOCSolver(A, B, f, initial_position, initial_position, horizon, max_speed)

    def set_desired_state(self, state):
        """ set xbar of solver """
        self.solver.set_xbar(state)

    def update_then_calculate_optimal_actuation(self, current_position):
        """
        calculates optimal input for controller
        :param current_position: current position
        :return: optimal velocity command
        """
        return self.solver.calculate_optimal_actuation(current_position)


class TestCFTOCSolver(unittest.TestCase):
    def test_basic_controller(self):

        for soft in [False, True]:

            step_time = 1

            robot_position = np.zeros((3,))
            max_speed = 0.2 / 3

            robot_controller = PositionMPCController2D(step_time=step_time, max_speed=max_speed, soft=soft)

            tolerance = 0.01

            for t in range(5):
                robot_velocity = robot_controller.update_then_calculate_optimal_actuation(robot_position)
                assert (
                    np.linalg.norm(robot_velocity, np.inf) < max_speed + tolerance
                ), "Robot controller violating constraints"
                robot_position += robot_velocity * step_time

            assert np.linalg.norm(robot_position) < tolerance, "Robot unable to hold a zero position"

            tracking_position = np.array([1, 1, 0])
            robot_controller.set_desired_state(tracking_position)

            for t in range(20):
                robot_velocity = robot_controller.update_then_calculate_optimal_actuation(robot_position)
                assert (
                    np.linalg.norm(robot_velocity, np.inf) < max_speed + tolerance
                ), "Robot controller violating constraints"
                robot_position += robot_velocity * step_time

            assert np.linalg.norm(robot_position - tracking_position) < tolerance, "Robot unable to hold a new position"

            tracking_position = np.array([0, 0, 1])
            robot_controller.set_desired_state(tracking_position)

            for t in range(20):
                robot_velocity = robot_controller.update_then_calculate_optimal_actuation(robot_position)
                assert (
                    np.linalg.norm(robot_velocity, np.inf) < max_speed + tolerance
                ), "Robot controller violating constraints"
                robot_position += robot_velocity * step_time

            assert np.linalg.norm(robot_position - tracking_position) < tolerance, "Robot unable to hold a new position"

            if soft:
                print("Soft Constraint Controller passed unit tests")
            else:
                print("Hard Constraint Controller passed unit tests")


if __name__ == "__main__":
    unittest.main()
