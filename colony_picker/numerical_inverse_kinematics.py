import numpy as np
import math
import matplotlib.pyplot as plt
import pyvista as pv
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from functools import partial
import logging

# Denavit-Hartenberg Parameters of AR3 provided by
# AR2 Version 2.0 software executable files from
# https://www.anninrobotics.com/downloads
# parameters are the same between the AR2 and AR3
alphas = [-(math.pi/2), 0, math.pi/2, -(math.pi/2), math.pi/2, 0]
a_vals = [64.2, 305, 0, 0, 0, 0]
d_vals = [169.77, 0, 0, -222.63, 0, -36.25]
theta_offsets = [0, 0, -math.pi/2, 0, 0, math.pi]

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
        accumulator_t_mat = np.matmul(accumulator_t_mat, t_mat)
        t_mats.append(accumulator_t_mat)

    return t_mats


def draw_coordinate_system(plotter, t_mat, base=False, name=""):
    colors = ["red", "green", "blue"]
    position = t_mat[:3, 3]
    sphere = pv.Sphere(center=position, radius=4)
    if base:
        sphere_color = "black"
    else:
        sphere_color = "yellow"
    plotter.add_mesh(sphere, color=sphere_color, name=(name + "_sphere"))
    for axis_idx in range(3):
        axis = pv.Arrow(
            start=position, direction=t_mat[:3, axis_idx], scale=50)
        plotter.add_mesh(axis, color=colors[axis_idx], name=(
            name + f"_coord_{axis_idx}"))


def draw_links(plotter, joint_positions):
    """Given an array of joint, draw the links between each joint"""
    poly = pv.PolyData()
    poly.points = joint_positions
    cells = np.full((len(joint_positions)-1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(joint_positions)-1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(joint_positions), dtype=np.int_)
    poly.lines = cells
    poly["scalars"] = np.arange(poly.n_points)
    tube = poly.tube(radius=10)
    plotter.add_mesh(tube, color="black", name="robot skeleton", opacity=0.9)


# figured out which joint to end effector vectors to use from this link: https://automaticaddison.com/the-ultimate-guide-to-jacobian-matrices-for-robotics/


def get_joint_to_end_effector_vectors(t_mats):
    joint_to_end_effector_vectors = []
    end_effector_t_mat = t_mats[-1]
    for joint_idx in range(NUM_JOINTS):
        t_mat = t_mats[joint_idx]
        joint_to_end_effector_vector = np.subtract(
            end_effector_t_mat[:3, 3], t_mat[:3, 3])
        joint_to_end_effector_vectors.append(joint_to_end_effector_vector)
    return joint_to_end_effector_vectors


def draw_joint_to_end_effector_vector(plotter, t_mat, end_effector_pos):
    points = np.array(
        [t_mat[:3, 3], end_effector_pos])
    draw_links(plotter, points)

# Got the formula for the rotational component of the Jacobian from this website:
# https://www.rosroboticslearning.com/jacobian
# Got the formula for the cross product method for computing the translational
# component of the Jacobian from slide 19/30 of this
# lecture: https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/IK.pdf


def get_jacobian(thetas):
    t_mats = get_t_mats(thetas)
    jacobian = np.zeros((6, NUM_JOINTS))
    base_t_mat = np.identity(4)
    t_mats.insert(0, base_t_mat)
    end_effector_vectors = get_joint_to_end_effector_vectors(t_mats)
    for joint_idx in range(NUM_JOINTS):
        t_mat = t_mats[joint_idx]
        rot_axis = t_mat[:3, 2]
        jacobian[:3, joint_idx] = np.cross(
            rot_axis, end_effector_vectors[joint_idx])
        jacobian[3:, joint_idx] = rot_axis
    return jacobian


def get_euler_from_rot_mat(rot_mat):
    r = Rotation.from_matrix(rot_mat)
    return r.as_euler("xyz")


def get_end_effector_pose(thetas):
    t_mats = get_t_mats(thetas)
    end_effector_position = t_mats[-1][:3, 3]
    rot_mat = t_mats[-1][:3, :3]
    end_effector_rotation = get_euler_from_rot_mat(rot_mat)
    end_effector_pose = np.zeros(6)
    end_effector_pose[:3] = end_effector_position
    end_effector_pose[3:] = end_effector_rotation
    return end_effector_pose


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


def get_shortest_angle_to_target_in_degrees(target_angle, source_angle):
    target_angle = target_angle*(math.pi/180)
    source_angle = source_angle*(math.pi/180)
    return get_shortest_angle_to_target_in_radians(target_angle, source_angle)*(180/math.pi)


def get_directional_error(desired_end_effector_pose, current_end_effector_pose):
    directional_error = np.subtract(
        desired_end_effector_pose, current_end_effector_pose)
    for axis_idx in range(3):
        directional_rotational_error = get_shortest_angle_to_target_in_radians(
            desired_end_effector_pose[3 + axis_idx], current_end_effector_pose[3 + axis_idx])
        directional_error[3 + axis_idx] = directional_rotational_error
    return directional_error

# confirmed formula using this: https://stackoverflow.com/questions/15927755/opposite-of-numpy-unwrap


def joint_angles_to_euler(joint_angles, radians=True):
    if radians:
        period = np.pi
    else:
        period = 180
    return (joint_angles + period) % (2 * period) - period


def objective_function(thetas, desired_end_effector_pose):
    current_end_effector_pose = get_end_effector_pose(thetas)
    error_vector = get_directional_error(
        desired_end_effector_pose, current_end_effector_pose)

    error = np.linalg.norm(error_vector)
    return error


def get_gradient(thetas, desired_end_effector_pose):
    current_end_effector_pose = get_end_effector_pose(thetas)
    error_vector = get_directional_error(
        desired_end_effector_pose, current_end_effector_pose)
    jacobian = get_jacobian(thetas)
    jacobian_generalized_inverse = np.linalg.pinv(jacobian)
    print(np.any(np.abs(error_vector) > 10))
    d_thetas = -np.matmul(jacobian_generalized_inverse, error_vector)
    return d_thetas


def scipy_find_joint_angles(thetas_init, desired_end_effector_pose):

    # error = objective_function(thetas_init.copy(), desired_end_effector_pose)
    # thetas = thetas_init
    # for i in range(100):
    #     grad = get_gradient(thetas, desired_end_effector_pose)

    #     thetas += grad * 0.000000000001
    #     error = objective_function(thetas, desired_end_effector_pose)
    res = minimize(objective_function, thetas_init,
                   (desired_end_effector_pose,), method='CG', jac={'3-point'})
    print(res)
    return res.x


def find_joint_angles(current_thetas, desired_end_effector_pose):
    current_end_effector_pose = get_end_effector_pose(current_thetas)

    error_directional = get_directional_error(
        desired_end_effector_pose, current_end_effector_pose)

    error = np.linalg.norm(error_directional)
    iters = 0
    desired_error = np.zeros(6)
    desired_error += 1e-5
    step_decay_rate = 0.5

    errors = []
    for end_effector_idx in range(6):
        errors.append([])

    step_size = 1
    last_thetas = current_thetas
    while np.any(np.greater(error, desired_error)) and iters < 1000:

        current_end_effector_pose = get_end_effector_pose(current_thetas)

        error_directional = get_directional_error(
            desired_end_effector_pose, current_end_effector_pose)
        new_error = np.linalg.norm(error_directional)
        if new_error > error:
            current_thetas = last_thetas
            step_size *= step_decay_rate
        else:
            # step_size *= 1.1
            last_thetas = current_thetas
            error = new_error
        # print(f"=======ITER {iters}=======")
        # print(f"ERROR: {error_directional}")
        # print(f"CURRENT END EFFECTOR POSE: {current_end_effector_pose}")
        # print(f"CURRENT JOINT ANGLES: {current_thetas}")
        # print(f"=========================")

        for end_effector_idx in range(6):
            errors[end_effector_idx].append(
                error_directional[end_effector_idx])

        jacobian = get_jacobian(current_thetas)
        jacobian_generalized_inverse = np.linalg.pinv(jacobian)

        d_thetas = np.matmul(jacobian_generalized_inverse,
                             step_size*error_directional)
        current_thetas = np.add(current_thetas, d_thetas)

        iters += 1
    if np.any(np.greater(error, desired_error)):
        print(iters)
        logging.warning("Inverse kinematics did not converge")

    # current_end_effector_pose[3:] = np.rad2deg(current_end_effector_pose[3:])
    print(f"*************************")
    print(f"final NUMBER OF ITERATIONS: {iters}")
    print(f"final ERROR: {error}")
    print(f"final END EFFECTOR POSE: {current_end_effector_pose}")
    print(f"final JOINT ANGLES: {current_thetas}")
    print(f"*************************")
    # labels = ["x", "y", "z", "x_rot", "y_rot", "z_rot"]
    # for end_effector_idx in range(6):
    #     plt.plot(list(range(iters)),
    #              errors[end_effector_idx], label=labels[end_effector_idx])
    # plt.legend()
    # plt.show()
    return current_thetas


def animate_inverse_kinematics_sphere():
    positions = np.zeros((NUM_JOINTS + 1, 3))
    p = pv.Plotter()
    draw_coordinate_system(p, np.eye(4), base=True)
    thetas_init = np.zeros(6)
    desired_end_effector_pose = np.array(get_end_effector_pose(thetas_init))

    def callback(idx, desired_end_effector_param):
        desired_end_effector_pose[idx] = desired_end_effector_param
        thetas = scipy_find_joint_angles(
            thetas_init, desired_end_effector_pose)
        thetas_init[:] = thetas
        t_mats = get_t_mats(thetas)
        for i, t_mat in enumerate(t_mats):
            positions[i + 1, :3] = t_mat[:3, 3]
            if i == NUM_JOINTS - 1:
                draw_coordinate_system(p, t_mat, name=f"theta{i}", base=True)
            else:
                draw_coordinate_system(p, t_mat, name=f"theta{i}")
        draw_links(p, positions)

    p.add_sphere_widget(
        partial(callback, range(0, 3)), center=desired_end_effector_pose[:3], radius=10, color="#00FF00", test_callback=False)
    end_effector_pose_labels = ["x_rot", "y_rot", "z_rot"]
    for i in range(3):
        p.add_slider_widget(partial(callback, i + 3),
                            [-np.pi, np.pi], value=desired_end_effector_pose[i + 3], pointa=(
            0.7, 0.9-0.15*i),
            pointb=(0.95, 0.9-0.15*i),
            title=end_effector_pose_labels[i], event_type="end")

    p.show()


def animate_inverse_kinematics_sliders():
    p = pv.Plotter()
    draw_coordinate_system(p, np.eye(4), base=True)
    thetas_init = np.zeros(6)
    desired_end_effector_pose = np.array(get_end_effector_pose(thetas_init))

    def callback(idx, desired_end_effector_pose_component):
        desired_end_effector_pose[idx] = desired_end_effector_pose_component
        thetas = find_joint_angles(thetas_init, desired_end_effector_pose)
        t_mats = get_t_mats(thetas)
        for i, t_mat in enumerate(t_mats):
            if i == NUM_JOINTS - 1:
                draw_coordinate_system(p, t_mat, name=f"theta{i}", base=True)
            else:
                draw_coordinate_system(p, t_mat, name=f"theta{i}")
    end_effector_pose_labels = ["x", "y", "z", "x_rot", "y_rot", "z_rot"]
    for i in range(6):
        if i <= 2:
            lower_bound = -700
            upper_bound = -lower_bound
        else:
            lower_bound = -np.pi
            upper_bound = -lower_bound
        p.add_slider_widget(partial(callback, (i,)),
                            [lower_bound, upper_bound], value=desired_end_effector_pose[i], pointa=(
                                0.7, 0.9-0.15*i),
                            pointb=(0.95, 0.9-0.15*i),
                            title=end_effector_pose_labels[i], event_type="end")

    p.show()


def animate_forward_kinematics():
    p = pv.Plotter()
    draw_coordinate_system(p, np.eye(4), True)
    thetas = np.zeros(6)

    def callback(idx, theta):
        thetas[idx] = theta
        t_mats = get_t_mats(thetas)
        for i, t_mat in enumerate(t_mats):
            draw_coordinate_system(p, t_mat, name=f"theta_{i}")

    for i in range(NUM_JOINTS):
        p.add_slider_widget(partial(callback, (i,)),
                            [-np.pi, np.pi], pointa=(0.7, 0.9-0.15*i),
                            pointb=(0.95, 0.9-0.15*i),
                            title=f"Theta {i}", event_type="always")
    p.show()


def test_plane_widget():
    p = pv.Plotter()
    test_geom = pv.Sphere()
    p.add_mesh(test_geom)

    def callback(normal, origin):
        print("NORMAL: ")
        print(normal)
        test_geom = pv.Sphere()
        p.add_mesh(test_geom, name="sphere", opacity=0.5)

    p.add_plane_widget(callback)
    p.show()


if __name__ == "__main__":
    # np.set_printoptions(precision=2, suppress=True)
    # thetas = np.ones(6)
    # thetas[5] = math.pi
    # thetas[1] = math.pi/2
    # get_jacobian(thetas)
    # print("Jacobian: ")
    # print(get_jacobian(thetas))

    thetas_init = np.zeros(NUM_JOINTS)
    # thetas_init = np.array([0.0001745329252,
    #                         -1.570796327,
    #                         0,
    #                         0.0001745329252,
    #                         0.0001745329252,
    #                         3.141767187])
    # print(thetas_init)
    # t_mats = get_t_mats(thetas_init)
    # print(len(t_mats))
    # for t_mat in t_mats:
    #     print(t_mat)

    # thetas = np.array([1, 2, 3, 4, 5, 6])*(math.pi/180)
    # thetas_init = thetas
    desired_end_effector_pose = np.array(
        [6.22080000e+02, 0,  1.70770000e+02, -3.14159265e+00, 1.57079631e+00, 0.00000000e+00])
    thetas_init = np.zeros(6)
    # print(find_joint_angles(np.zeros(6) + 1e-2,
    #       get_end_effector_pose(np.zeros(6))))
    # thetas_init = [-1.12056808, -0.1700862265, -1.195606121, -0.1360689812, -0.8864493921, 2.732385203]
    # scipy_find_joint_angles(thetas_init, desired_end_effector_pose)
    # animate_forward_kinematics()
    animate_inverse_kinematics_sphere()
    # animate_inverse_kinematics_sliders()
    # test_plane_widget()

    # display_robot_arm(get_t_mats(np.zeros(6)))
    # t_mats = get_t_mats(thetas)
    # print("T_MATS: ")
    # for t_mat in t_mats:
    #     print(t_mat)
    #     print()
    # print(get_end_effector_pose(thetas))
