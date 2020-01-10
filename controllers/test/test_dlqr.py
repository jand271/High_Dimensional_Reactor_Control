import numpy as np
import unittest
from controllers.dlqr import *

""" Model Dynamics """
step_time = 1
A = np.eye(3)  # state dynamics
B = step_time * np.eye(3)  # input velocity dynamics


class TestDLQRController(unittest.TestCase):

    def test_main(self):
        step_time = 1

        robot_position = np.ones((3,))

        robot_controller = DLQR(A, B)

        tolerance = 0.01

        for t in range(20):
            robot_velocity = robot_controller.update_then_calculate_optimal_actuation(robot_position)
            robot_position += robot_velocity * step_time

        assert np.linalg.norm(robot_position) < tolerance, "Robot unable to actuate a zero position"


class TestAffineDLQRController(unittest.TestCase):

    def test_main(self):
        step_time = 1

        robot_position = np.ones((3,))

        f = np.ones((3,))
        robot_controller = AffineDLQR(A, B, f)

        tolerance = 0.01

        for t in range(20):
            robot_velocity = robot_controller.update_then_calculate_optimal_actuation(robot_position)
            robot_position += robot_velocity * step_time + f

        assert np.linalg.norm(robot_position) < tolerance, "Robot unable to actuate a zero position"


class TestTrackingDLQRController(unittest.TestCase):

    def test_main(self):
        step_time = 1

        robot_position = np.zeros((3,))

        robot_controller = TrackingDLQR(A, B)

        tolerance = 0.01

        for t in range(5):
            robot_velocity = robot_controller.update_then_calculate_optimal_actuation(robot_position)
            robot_position += robot_velocity * step_time

        assert np.linalg.norm(robot_position) < tolerance, "Robot unable to hold a zero position"

        tracking_position = np.array([1, 1, 0])
        robot_controller.set_desired_state(tracking_position)

        for t in range(20):
            robot_velocity = robot_controller.update_then_calculate_optimal_actuation(robot_position)
            robot_position += robot_velocity * step_time

        assert np.linalg.norm(robot_position - tracking_position) < tolerance, "Robot unable to hold a new position"

        tracking_position = np.array([0, 0, 1])
        robot_controller.set_desired_state(tracking_position)

        for t in range(20):
            robot_velocity = robot_controller.update_then_calculate_optimal_actuation(robot_position)
            robot_position += robot_velocity * step_time

        assert np.linalg.norm(robot_position - tracking_position) < tolerance, "Robot unable to hold a new position"


class TestTrackingAffineDLQRController(unittest.TestCase):

    def test_main(self):
        step_time = 1

        robot_position = np.zeros((3,))

        f = np.ones((3,))
        robot_controller = TrackingAffineDLQR(A, B, f)

        tolerance = 0.01

        for t in range(20):
            robot_velocity = robot_controller.update_then_calculate_optimal_actuation(robot_position)
            robot_position += robot_velocity * step_time + f

        assert np.linalg.norm(robot_position) < tolerance, "Robot unable to hold a zero position"

        tracking_position = np.array([1, 1, 0])
        robot_controller.set_desired_state(tracking_position)

        for t in range(20):
            robot_velocity = robot_controller.update_then_calculate_optimal_actuation(robot_position)
            robot_position += robot_velocity * step_time + f

        assert np.linalg.norm(robot_position - tracking_position) < tolerance, "Robot unable to hold a new position"

        tracking_position = np.array([0, 0, 1])
        robot_controller.set_desired_state(tracking_position)

        for t in range(20):
            robot_velocity = robot_controller.update_then_calculate_optimal_actuation(robot_position)
            robot_position += robot_velocity * step_time + f

        assert np.linalg.norm(robot_position - tracking_position) < tolerance, "Robot unable to hold a new position"


if __name__ == '__main__':
    unittest.main()
