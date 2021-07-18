import numpy as np
import sympy as sp

# Denavit-Hartenberg Parameters of AR3 provided by
# AR2 Version 2.0 software executable files from
# https://www.anninrobotics.com/downloads
# parameters are the same between the AR2 and AR3
alphas = [-(sp.pi/2), 0, 90, -(sp.pi/2), sp.pi/2, 0]
a_vals = [64.2, 305, 0, 0, 0, 0]
d_vals = [169.77, 0, 0, -222.63, 0, -36.25]
theta_offsets = [0, 0, -90, 0, 0, 180]

NUM_JOINTS = 6


def get_t_mat(theta, theta_offset, a, alpha, d):
    theta += theta_offset
    return sp.Matrix([[sp.cos(theta),
                       -sp.sin(theta)*sp.cos(alpha),
                       sp.sin(theta)*sp.sin(alpha),
                       a*sp.cos(theta)],
                      [sp.sin(theta),
                          sp.cos(theta)*sp.cos(alpha),
                          -sp.cos(theta)*sp.sin(alpha),
                          a*sp.sin(theta)],
                      [0, sp.sin(alpha), sp.cos(alpha), d],
                      [0, 0, 0, 1]])


def get_end_effector_t_mat():
    thetas_str = ""
    for joint_idx in range(0, NUM_JOINTS):
        if joint_idx == NUM_JOINTS - 1:
            thetas_str += f"theta_{joint_idx}"
        else:
            thetas_str += f"theta_{joint_idx + 1}, "
    thetas = sp.symbols(thetas_str)
    end_effector_t_mat = sp.eye(4)
    for joint_idx in range(0, NUM_JOINTS):
        t_mat = get_t_mat(thetas[joint_idx],
                          theta_offsets[joint_idx],
                          a_vals[joint_idx],
                          alphas[joint_idx],
                          d_vals[joint_idx])
        end_effector_t_mat = t_mat * end_effector_t_mat
    return end_effector_t_mat


def get_jacobian():

    return 0


def get_jacobian_pseudoinverse(jacobian):
    return 0

# def find_joint_angles(desired_end_effector_pose):
#     current_end_effector_pose = np.zeros(6)
#     error = 100

#     while error > 0.01:
#         error = desired_end_effector_pose - current_end_effector_pose

#         jacobian = get_jacobian()
#         jacobian_pseudoinverse = get_jacobian_pseudoinverse(jacobian)
#         thetas += np.matmul(jacobian_pseudoinverse, desired_end_effector_pose)
