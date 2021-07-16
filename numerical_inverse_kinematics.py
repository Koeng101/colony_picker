import numpy as np
import math

# Denavit-Hartenberg Parameters of AR3 provided by
# AR2 Version 2.0 software executable files from
# https://www.anninrobotics.com/downloads
# parameters are the same between the AR2 and AR3
alphas = [-(math.pi/2), 0, math.pi, -(math.pi/2), math.pi/2, 0]
a_vals = [64.2, 305, 0, 0, 0, 0]
d_vals = [169.77, 0, 0, -222.63, 0, -36.25]
theta_offsets = [0, 0, -(math.pi/2), 0, 0, math.pi]

NUM_JOINTS = 6

# Got this formula for the transformation matrices using DH parameters
# from this site: https://blog.robotiq.com/how-to-calculate-a-robots-forward-kinematics-in-5-easy-steps


def get_t_mat(theta, theta_offset, a, alpha, d):
    theta += theta_offset
    return np.array([[math.cos(theta),
                      -math.sin(theta)*math.cos(alpha),
                      math.sin(theta)*math.sin(alpha),
                      a*math.cos(theta)],
                     [math.sin(theta),
                      math.cos(theta)*math.cos(alpha),
                      -math.cos(theta)*math.sin(alpha),
                      a*math.sin(theta)],
                     [0, math.sin(alpha), math.cos(alpha), d],
                     [0, 0, 0, 1]])


def get_t_mats(thetas):
    t_mats = []
    end_effector_t_mat = np.identity(4)
    for joint_idx in range(0, NUM_JOINTS):
        t_mat = get_t_mat(thetas[joint_idx],
                          theta_offsets[joint_idx],
                          a_vals[joint_idx],
                          alphas[joint_idx],
                          d_vals[joint_idx])
        t_mats.append(t_mat)
        end_effector_t_mat = np.matmul(t_mat, end_effector_t_mat)
        if joint_idx == NUM_JOINTS:
            t_mats.append(end_effector_t_mat)

    return t_mats


def get_joint_to_end_effector_vectors(t_mats):
    end_effector_vectors = []
    end_effector_t_mat = t_mats[-1]
    for joint_idx in range(0, NUM_JOINTS):
        t_mat = t_mats[joint_idx]
        end_effector_vector = np.subtract(
            end_effector_t_mat[:3, 3], t_mat[:3, 3])
        end_effector_vectors.append(end_effector_vector)
    return end_effector_vectors

# Got the formula for the rotational component of the Jacobian from this website:
# https://www.rosroboticslearning.com/jacobian
# Got the formula for the cross product method for computing the translational
# component of the Jacobian from slide 19/30 of this
# lecture: https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/IK.pdf


def get_jacobian(thetas):
    t_mats = get_t_mats(thetas)
    end_effector_vectors = get_joint_to_end_effector_vectors(t_mats)
    jacobian = np.zeros((6, NUM_JOINTS))
    for joint_idx in range(0, NUM_JOINTS):
        t_mat = t_mats[joint_idx]
        rot_axis = t_mat[:3, 2]
        jacobian[:3, joint_idx] = np.cross(
            rot_axis, end_effector_vectors[joint_idx])
        jacobian[3:, joint_idx] = rot_axis
    return jacobian

# used this source for this formula: https://stackoverflow.com/questions/15022630/how-to-calculate-the-angle-from-rotation-matrix


def get_euler_from_rot_mat(rot_mat):
    return np.array([math.atan2(rot_mat[2, 1], rot_mat[2, 2]),
                     math.atan2(-rot_mat[2, 0], math.sqrt(np.square(rot_mat[2, 1])
                                                          + np.square(rot_mat[2, 2]))),
                     math.atan2(rot_mat[1, 0], rot_mat[0, 0])])


def get_end_effector_pose(thetas):
    t_mats = get_t_mats(thetas)
    end_effector_position = t_mats[-1][:3, 3]
    rot_mat = t_mats[-1][:3, :3]
    end_effector_rotation = get_euler_from_rot_mat(rot_mat)
    end_effector_pose = np.zeros(6)
    end_effector_pose[:3] = end_effector_position
    end_effector_pose[3:] = end_effector_rotation
    return end_effector_pose


def find_joint_angles(current_thetas, desired_end_effector_pose):
    error = 1
    iters = 0
    while error > 0 and iters < 100:
        iters += 1
        current_end_effector_pose = get_end_effector_pose(current_thetas)
        error = np.subtract(
            desired_end_effector_pose, current_end_effector_pose)

        jacobian = get_jacobian(current_thetas)
        jacobian_generalized_inverse = np.linalg.pinv(jacobian)
        d_thetas = np.matmul(jacobian_generalized_inverse,
                             0.1*error)
        current_thetas = np.add(current_thetas, d_thetas)
        print(f"=======ITER {iter}=======")
        print(f"ERROR: {error}")
        print(f"CURRENT END EFFECTOR POSE: {current_end_effector_pose}")
        print(f"CURRENT JOINT ANGLES: {current_thetas}")
        print(f"=========================")


thetas_init = np.array([0, 0, 0, 0, 0, 0])
desired_end_effector_pose = np.array([0, 10, 20, 0, math.pi/2, 0])
find_joint_angles(thetas_init, desired_end_effector_pose)
