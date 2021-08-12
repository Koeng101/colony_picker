import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import math
from scipy.spatial.transform import Rotation


# Denavit-Hartenberg Parameters of AR3 provided by
# AR2 Version 2.0 software executable files from
# https://www.anninrobotics.com/downloads
# parameters are the same between the AR2 and AR3
alphas = [-(math.pi/2), 0, math.pi/2, -(math.pi/2), math.pi/2, 0]
a_vals = [64.2, 305, 0, 0, 0, 0]
d_vals = [169.77, 0, 0, -222.63, 0, -36.25]
theta_offsets = [0, 0, -math.pi/2, 0, 0, math.pi]

NUM_JOINTS = 6


def get_relative_t_mat(theta, theta_offset, a, alpha, d):
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


def get_theta_names():
    theta_names_str = ""
    for joint_idx in range(NUM_JOINTS):
        if joint_idx == NUM_JOINTS - 1:
            theta_names_str += f"theta_{joint_idx}"
        else:
            theta_names_str += f"theta_{joint_idx}, "
    theta_names = sp.symbols(theta_names_str)
    return theta_names


def get_unevaluated_t_mats(theta_names):
    accumulator_t_mat = sp.eye(4)
    unevaluated_t_mats = []
    for joint_idx in range(NUM_JOINTS):
        unevaluated_t_mat = get_relative_t_mat(theta_names[joint_idx],
                                               theta_offsets[joint_idx],
                                               a_vals[joint_idx],
                                               alphas[joint_idx],
                                               d_vals[joint_idx])
        accumulator_t_mat = accumulator_t_mat * unevaluated_t_mat
        unevaluated_t_mats.append(accumulator_t_mat)
    return unevaluated_t_mats


def get_unevaluated_jacobian(theta_names, unevaluated_t_mats):
    end_effector_position = unevaluated_t_mats[-1][:3, 3]
    unevaluated_jacobian = sp.zeros(6, NUM_JOINTS)
    base_t_mat = sp.eye(4)
    unevaluated_t_mats.insert(0, base_t_mat)
    for joint_idx in range(NUM_JOINTS):
        unevaluated_t_mat = unevaluated_t_mats[joint_idx]
        rotation_axis = unevaluated_t_mat[:3, 2]
        unevaluated_jacobian[3:, joint_idx] = rotation_axis
    for axis_idx in range(3):
        end_effector_position_expression = end_effector_position[axis_idx]
        for joint_idx in range(NUM_JOINTS):
            unevaluated_jacobian[axis_idx, joint_idx] = sp.diff(
                end_effector_position_expression, theta_names[joint_idx])
    return unevaluated_jacobian


def get_theta_substitutions(theta_names, theta_vals):
    theta_substitutions = []
    for joint_idx in range(NUM_JOINTS):
        theta_substitutions.append(
            (theta_names[joint_idx], theta_vals[joint_idx]))
    return theta_substitutions


def get_jacobian(theta_substitutions, unevaluated_jacobian):
    jacobian = unevaluated_jacobian.subs(theta_substitutions)
    return np.array(jacobian).astype(np.float64)


# def get_euler_from_rot_mat(rot_mat):
#     return np.array([math.atan2(rot_mat[2, 1], rot_mat[2, 2]),
#                      math.atan2(-rot_mat[2, 0], math.sqrt(np.square(rot_mat[2, 1])
#                                                           + np.square(rot_mat[2, 2]))),
#                      math.atan2(rot_mat[1, 0], rot_mat[0, 0])])

def get_euler_from_rot_mat(rot_mat):
    r = Rotation.from_matrix(rot_mat)
    return r.as_euler("xyz")

# got this formula from here: https://stackoverflow.com/questions/1878907/how-can-i-find-the-difference-between-two-angles


def get_shortest_angle_to_target_in_radians(target_angle, source_angle):
    # returns directional angle
    a = target_angle - source_angle
    if a > math.pi:
        return a - 2*math.pi
    elif a < -math.pi:
        return a + 2*math.pi
    else:
        return a


def get_directional_error(desired_end_effector_pose, current_end_effector_pose):
    directional_error = np.subtract(
        desired_end_effector_pose, current_end_effector_pose)
    for axis_idx in range(3):
        directional_rotational_error = get_shortest_angle_to_target_in_radians(
            desired_end_effector_pose[3 + axis_idx], current_end_effector_pose[3 + axis_idx])
        directional_error[3 + axis_idx] = directional_rotational_error
    return directional_error


def get_end_effector_pose(theta_substitutions, unevaluated_t_mats):
    end_effector_t_mat = unevaluated_t_mats[-1].subs(theta_substitutions)
    end_effector_t_mat = np.array(end_effector_t_mat).astype(np.float64)
    end_effector_position = end_effector_t_mat[:3, 3]
    end_effector_rotation_mat = end_effector_t_mat[:3, :3]
    end_effector_rotation = get_euler_from_rot_mat(end_effector_rotation_mat)
    end_effector_pose = np.zeros(6)
    end_effector_pose[:3] = end_effector_position
    end_effector_pose[3:] = end_effector_rotation
    return end_effector_pose


def find_joint_angles(current_thetas, desired_end_effector_pose):
    error = np.ones(6)
    iters = 0
    desired_error = np.zeros(6)
    theta_names = get_theta_names()
    unevaluated_t_mats = get_unevaluated_t_mats(theta_names)
    unevaluated_jacobian = get_unevaluated_jacobian(
        theta_names, unevaluated_t_mats)
    # for plotting:
    errors = [[] for end_effector_idx in range(6)]

    while np.any(np.greater(error, desired_error)) and iters < 1000:
        current_theta_substitutions = get_theta_substitutions(
            theta_names, current_thetas)
        current_end_effector_pose = get_end_effector_pose(
            current_theta_substitutions, unevaluated_t_mats)

        error_directional = np.subtract(
            desired_end_effector_pose, current_end_effector_pose)
        error = np.absolute(error_directional)
        print(f"=======ITER {iters}=======")
        print(f"ERROR: {error_directional}")
        print(f"CURRENT END EFFECTOR POSE: {current_end_effector_pose}")
        print(f"CURRENT JOINT ANGLES: {current_thetas}")
        print(f"=========================")

        for end_effector_idx in range(6):
            errors[end_effector_idx].append(
                error_directional[end_effector_idx])

        jacobian = get_jacobian(
            current_theta_substitutions, unevaluated_jacobian)
        jacobian_generalized_inverse = np.linalg.pinv(jacobian)
        d_thetas = np.matmul(jacobian_generalized_inverse,
                             0.01*error_directional)
        current_thetas = np.add(current_thetas, d_thetas)
        iters += 1
    current_thetas = current_thetas*(180/np.pi)
    print(f"*************************")
    print(f"final NUMBER OF ITERATIONS: {iters}")
    print(f"final ERROR: {error}")
    print(f"final END EFFECTOR POSE: {current_end_effector_pose}")
    print(f"final JOINT ANGLES: {current_thetas}")
    print(f"*************************")

    for end_effector_idx in range(6):
        plt.plot(list(range(0, iters)), errors[end_effector_idx])
    plt.show()


# thetas_init = np.zeros(NUM_JOINTS)
# desired_end_effector_pose = np.array(
#     [564.4182487, 20.46563405, 63.83689114, 71.95398308*(math.pi/180), 6.263122866*(math.pi/180), 105.7251901*(math.pi/180)])
# find_joint_angles(thetas_init, desired_end_effector_pose)

theta_vals = np.array([-1.12056808,
                       - 0.1700862265,
                       -1.195606121,
                       -0.1360689812,
                       -0.8864493921,
                       2.732385203])

theta_names = get_theta_names()
unevaluated_t_mats = get_unevaluated_t_mats(theta_names)
unevaluated_jacobian = get_unevaluated_jacobian(
    theta_names, unevaluated_t_mats)
theta_substitutions = get_theta_substitutions(theta_names, theta_vals)
jacobian = get_jacobian(theta_substitutions, unevaluated_jacobian)
t_mats_list = []
for t_mat, t_mat_idx in zip(unevaluated_t_mats, range(len(unevaluated_t_mats))):
    new_t_mat = unevaluated_t_mats[t_mat_idx].subs(theta_substitutions)
    new_t_mat = np.array(new_t_mat).astype(np.float64)
    t_mats_list.append(new_t_mat)
print("SYMPY T MATS:")
for t_mat in t_mats_list:
    print(t_mat)
    print()
np.set_printoptions(precision=2, suppress=True)
print("SYMPY JACOBIAN: ")
print(jacobian)
print()
# end_effector_pose = get_end_effector_pose(
#     theta_substitutions, unevaluated_t_mats)
# print("SYMPY END EFFECTOR POSE: ")
# print(end_effector_pose)
