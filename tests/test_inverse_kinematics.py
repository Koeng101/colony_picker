"""Tests numerical inverse kinematics pipeline.
"""
import unittest
from colony_picker.numerical_inverse_kinematics import*
import random
import math

ar3_alpha_vals = [-(math.pi/2), 0, math.pi/2, -(math.pi/2), math.pi/2, 0]
ar3_a_vals = [64.2, 305, 0, 0, 0, 0]
ar3_d_vals = [169.77, 0, 0, -222.63, 0, -36.25]
ar3_theta_offsets = [0, 0, -math.pi/2, 0, 0, math.pi]
ar3_num_joints = len(ar3_alpha_vals)

ar3_dh_params = np.array(
    [ar3_theta_offsets, ar3_alpha_vals, ar3_a_vals, ar3_d_vals])


class TestInverseKinematics(unittest.TestCase):
    """A unit test harness for testing the numerical inverse kinematics pipeline
    """

    def test_wrap_joint_angles(self):
        test_cases = [(7, 7), (367, 7), (540, -180),
                      (-75, -75), (-360, 0), (-367, -7),
                      (-460, -100)]
        for test_case in test_cases:
            joint_angle, euler_angle = test_case
            joint_angle_arr = np.array([joint_angle])
            euler_angle_arr = np.array([euler_angle])
            self.assertTrue(wrap_joint_angles(
                joint_angle_arr, radians=False) == euler_angle_arr)

    def test_find_joint_angles(self):
        iters = 1000
        thetas_init = np.zeros(ar3_num_joints)
        joint_angles = np.array([[random.uniform(-math.pi, math.pi)
                                 for i in range(iters)] for i in range(ar3_num_joints)])
        for test_idx in range(iters):
            test_joint_angles = joint_angles[:, test_idx]
            target_pose = get_end_effector_pose(
                test_joint_angles, ar3_dh_params)
            _, solved = find_joint_angles(
                thetas_init, target_pose, ar3_dh_params)
            self.assertTrue(solved)

    @unittest.skip("No reason")
    def test_animate_plot(self):
        animate_forward_kinematics(ar3_dh_params)

    @unittest.skip("No reason")
    def test_animate_inverse_kinematics_sphere(self):
        animate_inverse_kinematics_sphere(ar3_dh_params)

    @unittest.skip("No reason")
    def test_animate_inverse_kinematics_sliders(self):
        animate_inverse_kinematics_sliders(ar3_dh_params)


if __name__ == "__main__":
    unittest.main()
