import numpy as np
import math
import matplotlib.pyplot as plt
import pyvista as pv

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
    accumulator_t_mat = np.identity(4)
    for joint_idx in range(NUM_JOINTS):
        t_mat = get_t_mat(thetas[joint_idx],
                          theta_offsets[joint_idx],
                          a_vals[joint_idx],
                          alphas[joint_idx],
                          d_vals[joint_idx])
        accumulator_t_mat = np.matmul(t_mat, accumulator_t_mat)
        t_mats.append(accumulator_t_mat)
    display_joints(t_mats)
    
    return t_mats

def polyline_from_points(points):
    poly = pv.PolyData()
    poly.points = points
    the_cell = np.arange(0, len(points), dtype=np.int_)
    the_cell = np.insert(the_cell, 0, len(points))
    poly.lines = the_cell
    return poly

def display_joints(t_mats):
    plotter = pv.Plotter()
    colors = ["red", "green", "blue", "purple", "pink", "yellow"]
    points = 
    for joint_idx in range(NUM_JOINTS):
        t_mat = t_mats[joint_idx]
        sphere = pv.Sphere(radius = 100, center = t_mat[:3,3])
        arrow = pv.Arrow()
        plotter.add_mesh(sphere, color = colors[joint_idx])

    plotter.show()
    

# Error: assumed all transformation matrices are in the base frame when we must
# multiply them to get them as such


def get_joint_to_end_effector_vectors(t_mats):
    end_effector_vectors = []
    end_effector_t_mat = t_mats[-1]
    for joint_idx in range(NUM_JOINTS):
        t_mat = t_mats[joint_idx]
        joint_to_end_effector_vector = np.subtract(
            end_effector_t_mat[:3, 3], t_mat[:3, 3])
        end_effector_vectors.append(joint_to_end_effector_vector)
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
    for joint_idx in range(NUM_JOINTS):
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

# def get_euler_from_rot_mat(rot_mat):
#     sy = math.sqrt(rot_mat[0, 0] * rot_mat[0, 0] +
#                    rot_mat[1, 0] * rot_mat[1, 0])
#     singular = sy < 1e-6
#     if not singular:
#         x = math.atan2(rot_mat[2, 1], rot_mat[2, 2])
#         y = math.atan2(-rot_mat[2, 0], sy)
#         z = math.atan2(rot_mat[1, 0], rot_mat[0, 0])
#     else:
#         x = math.atan2(-rot_mat[1, 2], rot_mat[1, 1])
#         y = math.atan2(-rot_mat[2, 0], sy)
#         z = 0
#     return np.array([x, y, z])


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
    error = np.ones(6)
    iters = 0
    desired_error = np.zeros(6)

    errors = []
    for end_effector_idx in range(6):
        errors.append([])

    while np.any(np.greater(error, desired_error)) and iters < 1000:
        current_end_effector_pose = get_end_effector_pose(current_thetas)

        error_directional = np.subtract(
            desired_end_effector_pose, current_end_effector_pose)
        error = np.absolute(error_directional)
        # print(f"=======ITER {iters}=======")
        # print(f"ERROR: {error_directional}")
        # print(f"CURRENT END EFFECTOR POSE: {current_end_effector_pose}")
        # print(f"CURRENT JOINT ANGLES: {current_thetas}")
        # print(f"=========================")

        for end_effector_idx in range(6):
            errors[end_effector_idx].append(
                current_end_effector_pose[end_effector_idx])

        jacobian = get_jacobian(current_thetas)
        jacobian_generalized_inverse = np.linalg.pinv(jacobian)
        d_thetas = np.matmul(jacobian_generalized_inverse,
                             0.002*error_directional)
        current_thetas = np.add(current_thetas, d_thetas)
        iters += 1

    # print(f"*************************")
    # print(f"final NUMBER OF ITERATIONS: {iters}")
    # print(f"final ERROR: {error}")
    # print(f"final END EFFECTOR POSE: {current_end_effector_pose}")
    # print(f"final JOINT ANGLES: {current_thetas}")
    # print(f"*************************")

    for end_effector_idx in range(6):
        plt.plot(list(range(0, iters)), errors[end_effector_idx])
    plt.show()


# thetas_init = np.array([0, 0, 0, 0, 0, 0])
# desired_end_effector_pose = np.array([30, 0, 475, -math.pi, math.pi, -math.pi])
# find_joint_angles(thetas_init, desired_end_effector_pose)
# t_mats = get_t_mats(thetas_init)
# print(len(t_mats))
# for t_mat in t_mats:
#     print(t_mat)
