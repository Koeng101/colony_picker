import numpy as np
import math
import matplotlib.pyplot as plt
import pyvista as pv

# Denavit-Hartenberg Parameters of AR3 provided by
# AR2 Version 2.0 software executable files from
# https://www.anninrobotics.com/downloads
# parameters are the same between the AR2 and AR3
alphas = [-(math.pi/2), 0, math.pi/2, -(math.pi/2), math.pi/2, 0]
a_vals = [64.2, 305, 0, 0, 0, 0]
d_vals = [169.77, 0, 0, -222.63, 0, -36.25]
theta_offsets = [0, -(math.pi/2), 0, 0, 0, math.pi]

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


def get_euler_from_rot_mat(rot_mat):
    return np.array([math.atan2(rot_mat[2, 1], rot_mat[2, 2]),
                     math.atan2(-rot_mat[2, 0], math.sqrt(np.square(rot_mat[2, 1])
                                                          + np.square(rot_mat[2, 2]))),
                     math.atan2(rot_mat[1, 0], rot_mat[0, 0])])


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
    tube = poly.tube(radius=1)
    plotter.add_mesh(tube, color="yellow")


def display_robot_arm(t_mats):
    plotter = pv.Plotter()
    positions = np.zeros((NUM_JOINTS + 1, 3))
    t_mat = np.identity(4)
    draw_coordinate_system(plotter, t_mat, base=True)
    positions[0, :3] = t_mat[:3, 3]
    for joint_idx in range(NUM_JOINTS):
        t_mat = t_mats[joint_idx]
        draw_coordinate_system(plotter, t_mat)
        positions[joint_idx, :3] = t_mat[:3, 3]
    draw_links(plotter, positions)
    poly = pv.PolyData(positions[1:])
    poly["labels"] = [f"Joint {i}" for i in range(NUM_JOINTS)]
    plotter.add_point_labels(
        poly, "labels", point_size=10, font_size=16, always_visible=True)
    plotter.slider
    plotter.show()


def animate_robot_arm():
    from functools import partial
    p = pv.Plotter()
    draw_coordinate_system(p, np.eye(4), True)
    thetas = np.zeros(6)

    def callback(idx, theta):
        thetas[idx] = theta
        t_mats = get_t_mats(thetas)
        # end_effector_pos = t_mats[-1][:3, 3]

        for i, t_mat in enumerate(t_mats):
            draw_coordinate_system(p, t_mat, name=f"theta_{i}")
            # if i != 5:
            #     draw_joint_to_end_effector_vector(p, t_mat, end_effector_pos)

    for i in range(NUM_JOINTS):
        p.add_slider_widget(partial(callback, (i,)),
                            [-np.pi, np.pi], pointa=(0.7, 0.9-0.15*i),
                            pointb=(0.95, 0.9-0.15*i),
                            title=f"Theta {i}", event_type="always")
    p.show()


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
    end_effector_vectors = get_joint_to_end_effector_vectors(t_mats)
    jacobian = np.zeros((6, NUM_JOINTS))
    base_t_mat = np.identity(4)
    t_mats.insert(0, base_t_mat)
    for joint_idx in range(NUM_JOINTS):
        t_mat = t_mats[joint_idx]
        rot_axis = t_mat[:3, 2]
        jacobian[:3, joint_idx] = np.cross(
            rot_axis, end_effector_vectors[joint_idx])
        jacobian[3:, joint_idx] = rot_axis
    print("JACOBIAN:")
    print(jacobian)
    return jacobian

# used this source for this formula: https://stackoverflow.com/questions/15022630/how-to-calculate-the-angle-from-rotation-matrix


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

    while np.any(np.greater(error, desired_error)) and iters < 10000:
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
                error_directional[end_effector_idx])

        jacobian = get_jacobian(current_thetas)
        jacobian_generalized_inverse = np.linalg.pinv(jacobian)
        d_thetas = np.matmul(jacobian_generalized_inverse,
                             0.001*error_directional)
        current_thetas = np.add(current_thetas, d_thetas)
        iters += 1
    current_thetas = current_thetas*(180/math.pi)
    print(f"*************************")
    print(f"final NUMBER OF ITERATIONS: {iters}")
    print(f"final ERROR: {error}")
    print(f"final END EFFECTOR POSE: {current_end_effector_pose}")
    print(f"final JOINT ANGLES: {current_thetas}")
    print(f"*************************")

    for end_effector_idx in range(6):
        plt.plot(list(range(0, iters)), errors[end_effector_idx])
    plt.show()


np.set_printoptions(precision=1, suppress=True)
thetas = np.zeros(6)
print(get_jacobian(thetas))

# thetas_init = np.zeros(6)
# desired_end_effector_pose = np.array([323.08*(math.pi/180), 0, 474.77 *
#                                       (math.pi/180), -math.pi/2, math.pi/2, -math.pi/2])
# find_joint_angles(thetas_init, desired_end_effector_pose)
# t_mats = get_t_mats(thetas_init)
# print(len(t_mats))
# for t_mat in t_mats:
#     print(t_mat)
