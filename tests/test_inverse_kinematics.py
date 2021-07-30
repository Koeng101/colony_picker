"""Tests the pose submodule.
"""
import unittest
from typing import Union
import numpy as np
import math
import matplotlib
import pyvista
import quaternion
from colony_picker.numerical_inverse_kinematics import*


class test_inverse_kinematics(unittest.TestCase):
    """A unit test harness for the Pose class
    """

    @unittest.skip("No reason")
    def test_get_t_mats(self):
        """Tests whether get_t_mats successfully generates transformation matrices
        in the base frame for each arm joint.
        """

        thetas = [0 for joint in list(range(NUM_JOINTS))]

        t_mats = get_t_mats(thetas)
        self.assertTrue(len(t_mats) == NUM_JOINTS)

        for t_mat_idx_1 in range(len(t_mats)):
            for t_mat_idx_2 in range(t_mat_idx_1 + 1, len(t_mats)):
                self.assertTrue(not np.array_equal(
                    t_mats[t_mat_idx_1], t_mats[t_mat_idx_2]))

    @unittest.skip("No reason")
    def test_animate_plot(self):
        animate_robot_arm()

    def test_get_shortest_angle_to_target(self):
        test_cases = [(-30, 30, -60), (-30, 150, -180), (-30, -150, 120),
                      (-30, -50, 20), (30, 150, -120), (30, -150, 180),
                      (30, 50, -20), (150, -150, -60), (150, 170, -20),
                      (-170, -150, -20)]
        for test_case in test_cases:

            target_angle, source_angle, shortest_angle = test_case

            self.assertTrue(np.isclose(get_shortest_angle_to_target_in_degrees(
                target_angle, source_angle), shortest_angle))

            self.assertTrue(np.isclose(get_shortest_angle_to_target_in_degrees(
                source_angle, target_angle), -shortest_angle))

    def test_joint_angles_to_euler(self):
        test_cases = [(7, 7), (367, 7), (540, -180),
                      (-75, -75), (-360, 0), (-367, -7),
                      (-460, -100)]
        for test_case in test_cases:
            joint_angle, euler_angle = test_case
            joint_angle_arr = np.array([joint_angle])
            euler_angle_arr = np.array([euler_angle])
            self.assertTrue(joint_angles_to_euler(
                joint_angle_arr, radians=False) == euler_angle_arr)
    # def test_find_joint_angles(self):

    @unittest.skip("No reason")
    def test_directional_error(self):
        pos = np.zeros((30, 3))
        angs = np.linspace(0, 4*np.pi, len(pos))
        euler_angles = np.zeros((len(pos), 3))
        euler_angles[:, 0] = angs

        de_x = []
        for ea in euler_angles:
            curr_pose = np.zeros(6)
            des_pose = np.zeros(6)
            des_pose[3:] = ea
            de = get_directional_error(curr_pose, des_pose)
            de_x.append(np.abs(de[3]))

        plt.plot(angs, de_x)
        plt.show()


if __name__ == "__main__":
    unittest.main()
