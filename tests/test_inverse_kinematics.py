"""Tests the pose submodule.
"""
import unittest
from typing import Union
import numpy as np
import math
import matplotlib
import pyvista
from colony_picker.numerical_inverse_kinematics import*


class test_inverse_kinematics(unittest.TestCase):
    """A unit test harness for the Pose class
    """

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


if __name__ == "__main__":
    unittest.main()
