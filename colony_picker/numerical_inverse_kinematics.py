import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.lib.utils import source
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
    plotter.add_mesh(tube, color="black", name="robot skeleton", opacity=0.5)


def get_euler_from_rot_mat(rot_mat):
    r = Rotation.from_matrix(rot_mat)
    return r.as_euler("ZYX")


def get_quat_from_rot_mat(rot_mat):
    r = Rotation.from_matrix(rot_mat)
    return r.as_quat()


def convert_pose_to_quat(pose_in_euler):
    rot = Rotation.from_euler(
        'ZYX', pose_in_euler[3:], degrees=False)
    quat = rot.as_quat()
    pose_in_quat = np.zeros(7)
    pose_in_quat[:3] = pose_in_euler[:3]
    pose_in_quat[3:] = quat
    return pose_in_quat


def convert_pose_to_euler(pose_in_quat):
    rot = Rotation.from_quat(pose_in_quat[3:])
    euler = rot.as_euler("ZYX")
    pose_in_euler = np.zeros(6)
    pose_in_euler[:3] = pose_in_quat[:3]
    pose_in_euler[3:] = euler
    return pose_in_euler


def get_end_effector_pose(thetas, euler=False):
    t_mats = get_t_mats(thetas)
    end_effector_position = t_mats[-1][:3, 3]
    rot_mat = t_mats[-1][:3, :3]
    if euler:
        end_effector_rotation = get_euler_from_rot_mat(rot_mat)
        end_effector_pose = np.zeros(6)
    else:
        end_effector_rotation = get_quat_from_rot_mat(rot_mat)
        end_effector_pose = np.zeros(7)
    end_effector_pose[:3] = end_effector_position
    end_effector_pose[3:] = end_effector_rotation
    return end_effector_pose


def get_rotational_error(target_quat, source_quat):
    return np.arccos(np.clip(2*(np.dot(target_quat, source_quat)**2) - 1, -1, 1))


def get_error_vector(desired_end_effector_pose, current_end_effector_pose):
    error_vector = np.zeros(4)
    error_vector[:3] = np.subtract(
        desired_end_effector_pose[:3], current_end_effector_pose[:3])
    rotational_error = get_rotational_error(
        desired_end_effector_pose[3:], current_end_effector_pose[3:])
    error_vector[3] = rotational_error
    return error_vector


def get_mean_squared_error(error_vector):
    error_vector = error_vector**2
    error = np.sum(error_vector)
    error = error*(1/len(error_vector))
    return error


def objective_function(thetas, desired_end_effector_pose):
    current_end_effector_pose = get_end_effector_pose(thetas)
    error_vector = get_error_vector(
        desired_end_effector_pose, current_end_effector_pose)
    return get_mean_squared_error(error_vector)


def scipy_find_joint_angles(thetas_init, desired_end_effector_pose):
    result = minimize(objective_function, thetas_init,
                      (desired_end_effector_pose,), method='BFGS')
    if result.fun > 1e-07:
        print("ERROR TOO HIGH")
    print(result)
    return (result.x, result.fun)


def animate_inverse_kinematics_sliders():
    joint_positions = np.zeros((NUM_JOINTS + 1, 3))
    p = pv.Plotter()
    draw_coordinate_system(p, np.eye(4), base=True)
    thetas_init = np.zeros(6)
    desired_end_effector_pose = get_end_effector_pose(thetas_init, euler=True)

    def callback(idx, pose_param):
        desired_end_effector_pose[idx] = pose_param
        desired_end_effector_pose_quat = convert_pose_to_quat(
            desired_end_effector_pose)
        thetas, error = scipy_find_joint_angles(
            thetas_init, desired_end_effector_pose_quat)
        if error < 0.1:
            thetas_init[:] = thetas
        t_mats = get_t_mats(thetas)
        for i, t_mat in enumerate(t_mats):
            joint_positions[i + 1, :3] = t_mat[:3, 3]
            draw_coordinate_system(p, t_mat, name=f"theta{i}")
        draw_links(p, joint_positions)
    end_effector_pose_labels = ["x", "y", "z", "x_rot", "y_rot", "z_rot"]
    for i in range(6):
        if i <= 2:
            # TODO(ramininaieni): make the upper and lower bounds for position
            # more concrete
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


def animate_inverse_kinematics_sphere():
    joint_positions = np.zeros((NUM_JOINTS + 1, 3))
    p = pv.Plotter()
    draw_coordinate_system(p, np.eye(4), base=True)
    thetas_init = np.zeros(6)
    desired_end_effector_pose = get_end_effector_pose(thetas_init, euler=True)

    def callback(idx, pose_param):
        desired_end_effector_pose[idx] = pose_param
        desired_end_effector_pose_in_quat = convert_pose_to_quat(
            desired_end_effector_pose)
        thetas, error = scipy_find_joint_angles(
            thetas_init, desired_end_effector_pose_in_quat)
        if error < 0.1:
            thetas_init[:] = thetas

        t_mats = get_t_mats(thetas)
        for i, t_mat in enumerate(t_mats):
            joint_positions[i + 1, :3] = t_mat[:3, 3]
            draw_coordinate_system(p, t_mat, name=f"theta{i}")
        draw_links(p, joint_positions)

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


def animate_forward_kinematics():
    joint_positions = np.zeros((NUM_JOINTS + 1, 3))
    p = pv.Plotter()
    draw_coordinate_system(p, np.eye(4), True)
    thetas = np.zeros(6)

    def callback(idx, theta):
        thetas[idx] = theta
        t_mats = get_t_mats(thetas)
        for i, t_mat in enumerate(t_mats):
            joint_positions[i + 1, :3] = t_mat[:3, 3]
            draw_coordinate_system(p, t_mat, name=f"theta_{i}")
        draw_links(p, joint_positions)

    for i in range(NUM_JOINTS):
        p.add_slider_widget(partial(callback, (i,)),
                            [-np.pi, np.pi], pointa=(0.7, 0.9-0.15*i),
                            pointb=(0.95, 0.9-0.15*i),
                            title=f"Theta {i}", event_type="end")
    p.show()


if __name__ == "__main__":
    # animate_inverse_kinematics_sliders()
    animate_inverse_kinematics_sphere()
    # animate_forward_kinematics()
