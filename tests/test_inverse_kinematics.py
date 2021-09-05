"""Tests numerical inverse kinematics pipeline.
"""
import math
import unittest
import random
from colony_picker.inverse_kinematics import*
from tests.helper_functions_for_tests import*
from colony_picker.dh_params import AR3_DH_PARAMS, AR3_NUM_JOINTS

animation_test_warning = "Only one animation test should be run at a time."


class TestInverseKinematics(unittest.TestCase):
    """A unit test harness for testing the numerical inverse kinematics pipeline
    """

    def test_get_t_mat(self):
        """Tests that we can accurately compute the position and orientation
        of a single joint by comparing our solution to Chris Annin's solution
        for the AR3 robot arm given a single theta and set of dh params for that
        joint.
        """
        single_joint_dh_param = AR3_DH_PARAMS[:, 0]

        # Testing theta = 0 and thetas greater than 2*pi to confirm we can
        # handle angles greater than a period.
        test_0_solution = np.array([[1, 0, 0, 64.2], [0, 0, 1, 0], [
                                   0, -1, 0, 169.77], [0, 0, 0, 1]])
        self.assertTrue(
            np.allclose(get_t_mat(0, single_joint_dh_param), test_0_solution))
        self.assertTrue(
            np.allclose(get_t_mat(math.pi*2, single_joint_dh_param),
                        test_0_solution))
        self.assertTrue(
            np.allclose(get_t_mat(math.pi*4, single_joint_dh_param),
                        test_0_solution))

        # Testing theta = pi/2, confirming we can handle thetas other than zero
        # and that we can handle thetas within a quadrant.
        test_1_solution = np.array(
            [[0, 0, -1, 0], [1, 0, 0, 64.2], [0, -1, 0, 169.77], [0, 0, 0, 1]])
        self.assertTrue(
            np.allclose(get_t_mat(math.pi/2, single_joint_dh_param),
                        test_1_solution))

        # Testing theta = pi, confirming that we can handle the case when theta
        # is half a period and generally confirming we can handle thetas other
        # than zero.
        test_2_solution = np.array(
            [[-1, 0, 0, -64.2], [0, 0, -1, 0], [0, -1, 0, 169.77],
             [0, 0, 0, 1]])
        self.assertTrue(
            np.allclose(get_t_mat(math.pi, single_joint_dh_param),
                        test_2_solution))

    def test_get_t_mats(self):
        """Confirming that our forward kinematics pipeline results in the same
        values as those Chris Annin provided for the AR3 robot arm given
        different sets of thetas.
        """

        # Confirming we get the same result as Chris Annin when all thetas = 0.
        test_0_thetas = np.zeros(6)

        test_0_solution = np.array(
            [[0, 0, -1, 628.08], [0, -1, 0, 0], [-1, 0, 0, 169.77],
             [0, 0, 0, 1]])

        self.assertTrue(np.allclose(get_t_mats(
            test_0_thetas, AR3_DH_PARAMS)[-1], test_0_solution))

        # Confirming we get the same result as Chris Annin when all thetas are
        # not the same.
        test_1_thetas = np.array(
            [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi, 2*math.pi])

        test_1_solution = np.array(
            [[-0.5, 0.5, -0.7071067812, 148.0770064],
             [0.7071067812, 0.7071067812, 0, 0],
             [0.5, -0.5, -0.7071067812, -177.6881301], [0, 0, 0, 1]])

        # Confirming we get the same result as Chris Annin when most thetas are
        # greater than a period.
        self.assertTrue(np.allclose(get_t_mats(
            test_1_thetas, AR3_DH_PARAMS)[-1], test_1_solution))

        test_2_thetas = np.array([360, 405, 765, 450, 495, 540])*(math.pi/180)

        test_2_solution = np.array([[0, -1, 0, 279.8675683],
                                    [-0.7071067812, 0, 0.7071067812, -25.63262082],
                                    [-0.7071067812, 0, -0.7071067812, -242.8949474],
                                    [0, 0, 0, 1]])

        self.assertTrue(np.allclose(get_t_mats(
            test_2_thetas, AR3_DH_PARAMS)[-1], test_2_solution))

    def test_wrap_joint_angles(self):
        """Confirming that the wrap function properly confines angles in all
        quadrants that are greater than 360 degrees to a period of -180 -> 180
        degrees. 
        """
        # Testing angles in different quadrants and in different representations
        # (i.e. 367 and 7 which are the same angle, but both representations are
        # tested).
        test_cases = [(7, 7), (367, 7), (540, -180),
                      (-75, -75), (-360, 0), (-367, -7),
                      (-460, -100)]
        for test_case in test_cases:
            joint_angle, euler_angle = test_case
            joint_angle_arr = np.array([joint_angle])
            euler_angle_arr = np.array([euler_angle])
            self.assertTrue(wrap_joint_angles(
                joint_angle_arr, radians=False) == euler_angle_arr)

    def test_find_joint_angles_random_pose(self):
        """Testing that any random pose can be solved for with smart seed,
        regardless of the initial thetas fed into the optimization algorithm. 
        """
        iters = 100
        title_str = "find joint angles random pose"
        print_decorated_title(title_str)

        thetas_init = np.zeros(AR3_NUM_JOINTS)
        for iter in range(iters):
            # Confirming that optimization solves always given any reasonable
            # desired end effector pose.
            test_joint_angles = np.array(
                [random.uniform(-math.pi, math.pi) for i in range(AR3_NUM_JOINTS)])
            target_pose = get_end_effector_pose(
                test_joint_angles, AR3_DH_PARAMS)
            solved_joint_angles, solved = find_joint_angles(
                thetas_init, target_pose, AR3_DH_PARAMS, smart_seed=True)
            self.assertTrue(solved)

            # Confirming that optimization is solving for the correct thetas.
            solved_pose = get_end_effector_pose(
                solved_joint_angles, AR3_DH_PARAMS)
            self.assertTrue(np.allclose(get_error_vector(
                target_pose, solved_pose), np.zeros(4), atol=1e-02))
            draw_loading_bar(iter, iters, title_str)
        if PRINT_PROGRESS:
            print()

    def test_find_joint_angles_incremental_pose(self):
        """Testing that incrementally moving the end effector, by requiring the
        movement of a single joint, rather than choosing random positions, and
        seeding the optimization algorithm with the current thetas (instead of
        using smart seed), always solves.
        """
        iters = 100
        title_str = "find joint angles incremental pose"
        print_decorated_title(title_str)
        thetas_init = np.zeros(AR3_NUM_JOINTS)
        for iter in range(iters):
            rand_idx = random.randint(0, AR3_NUM_JOINTS - 1)
            rand_theta = random.uniform(-math.pi, math.pi)
            test_joint_angles = thetas_init
            # Choosing a set of thetas with only one theta value different
            # than the seed.
            test_joint_angles[rand_idx] = rand_theta
            target_pose = get_end_effector_pose(
                test_joint_angles, AR3_DH_PARAMS)
            solved_joint_angles, solved = find_joint_angles(
                thetas_init, target_pose, AR3_DH_PARAMS)

            self.assertTrue(solved)
            # Updating the seed to be the current thetas.
            thetas_init = solved_joint_angles

            draw_loading_bar(iter, iters, title_str)
        if PRINT_PROGRESS:
            print()

    @ unittest.skip(animation_test_warning)
    def test_animate_forward_kinematics(self):
        """Testing that each joint rotates around the previous joint's
        rotational axis when changing the corresponding theta and that this
        behavior is observed for every joint angle possible. Also testing that
        the structure of the arm matches the kinematics diagram provided by the
        manufacturer.
        """
        animate_forward_kinematics(AR3_DH_PARAMS)

    @ unittest.skip(animation_test_warning)
    def test_animate_inverse_kinematics_sphere(self):
        """Tests that when attempting impossible poses, the inverse kinematics
        pipeline does well at estimating the joint angles. Also tests that
        inverse kinematics solves for a variety of reasonable end effector poses
        that are both very close to the current pose and very far away.
        Human input allows us to test desired poses that are reasonable without
        needing to generate them with the forward kinematics pipeline first. 
        """
        animate_inverse_kinematics_sphere(AR3_DH_PARAMS)

    @ unittest.skip(animation_test_warning)
    def test_animate_inverse_kinematics_sliders(self):
        """Allows us to confirm that inverse kinematics results in the end
        effector moving as expected allong the x, y, and z axes as well as
        rotate in x, y, and z according to euler ZYX convention.
        """
        animate_inverse_kinematics_sliders(AR3_DH_PARAMS)


if __name__ == "__main__":
    unittest.main()
