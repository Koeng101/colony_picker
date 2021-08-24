import numpy as np
import math
import pyvista as pv
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from functools import partial
import random
import warnings
import time

# *************************************** #
#     INVERSE KINEMATICS CALCULATIONS     #
# *************************************** #


def get_euler_from_rot_mat(rot_mat: np.ndarray):
    """returns the ZYX euler angle represented by the inputted rotation matrix.

    Args:
        rot_mat (np.ndarray): a 3 x 3 rotation matrix.

    Returns:
        np.ndarray: a 1 x 3 array describing the rotation using the ZYX euler
            angle convention.
    """
    r = Rotation.from_matrix(rot_mat)
    return r.as_euler("ZYX")


def get_quat_from_rot_mat(rot_mat: np.ndarray):
    """returns the quaternion represented by the inputted rotation matrix.

    Args:
        rot_mat (np.ndarray): a 3 x 3 rotation matrix.

    Returns:
        np.ndarray: a 1 x 4 array describing the rotation as a quaternion.
    """
    r = Rotation.from_matrix(rot_mat)
    return r.as_quat()


def get_t_mat(theta: float, single_joint_dh_param: np.ndarray):
    """Uses the dh parameters of a single joint and its joint angle to output 
    the transformation matrix of that joint (its rotation and position) in the 
    previous joint's reference frame.

    Args:
        theta (float): the theta value of the single joint.
        single_joint_dh_param (np.ndarray): a 4 x 1 array containing the dh
            parameters to describe a single joint on a robot arm in the format
            [theta_offset, alpha, a, d].

    Returns:
        np.ndarray: a 4 x 4 transformation matrix describing the rotation and
            position of the inputted joint relative to the previous joint's
            reference frame.
    """
    theta_offset, alpha, a, d = single_joint_dh_param
    theta += theta_offset
    t_mat = np.array([[math.cos(theta),
                      -math.sin(theta)*math.cos(alpha),
                      math.sin(theta)*math.sin(alpha),
                      a*math.cos(theta)],
                      [math.sin(theta),
                      math.cos(theta)*math.cos(alpha),
                      -math.cos(theta)*math.sin(alpha),
                      a*math.sin(theta)],
                      [0, math.sin(alpha), math.cos(alpha), d],
                      [0, 0, 0, 1]])
    return t_mat


def get_t_mats(thetas: np.ndarray, dh_params: np.ndarray):
    """Conducts forward kinematics on a set of joint angles given dh parameters
    to describe the robot arm, producing a transformation matrix for each joint
    that describes the joint's position and orientation relative to the base
    frame. The last transformation matrix describes the position and orientation
    of the end effector relative to the base frame. 

    Args:
        thetas (np.ndarray): a 1 x n array of joint angles in the format
            [joint angle 1, joint angle 2, ... , joint angle n] in which n is 
            the number of joints on the robot arm. 
        dh_params (np.ndarray): a 4 x n array containing the dh parameters to 
            describe every joint on the robot arm in the format
            [[theta offset 1, theta offset 2, ... , theta offset n],
            [alpha 1, alpha 2, ... , alpha n],
            [a 1, a 2, ... , a n], 
            [d 1, d 2, ... , d n]]
            where n is the number of joints on the robot arm. 

    Returns:
        np.ndarray: an array containing 4 x 4 transformation matrices that each
            describe the position and orientation of each joint relative to the
            base frame in the format [joint 1, joint 2, ... , joint num joints]
    """
    num_joints = len(dh_params[0])
    t_mats = []
    accumulator_t_mat = np.identity(4)
    for joint_idx in range(num_joints):
        t_mat = get_t_mat(thetas[joint_idx], dh_params[:, joint_idx])
        accumulator_t_mat = np.matmul(accumulator_t_mat, t_mat)
        t_mats.append(accumulator_t_mat)

    return t_mats


def get_end_effector_pose(thetas: np.ndarray, dh_params: np.ndarray,
                          euler: bool = False):
    """Conducts forward kinematics on a robot arm described by dh_params,
    outputting the end effector pose given a set of joint angles described by
    thetas. 

    Args:
        thetas (np.ndarray): a 1 x n array of joint angles in the format
            [joint angle 1, joint angle 2, ... , joint angle n] in which n is 
            the number of joints on the robot arm. 
        dh_params (np.ndarray): a 4 x n array containing the dh parameters to 
            describe every joint on the robot arm in the format
            [[theta offset 1, theta offset 2, ... , theta offset n],
            [alpha 1, alpha 2, ... , alpha n],
            [a 1, a 2, ... , a n], 
            [d 1, d 2, ... , d n]]
            where n is the number of joints on the robot arm. 
        euler (bool, optional): whether the end effector pose should
            be returned with its orientation described in ZYX euler angles. 
            Defaults to False which implies orientation is described in
            quaternions.

    Returns:
        np.ndarray: an array describing the position and orientation of the end
            effector given thetas and dh_params. If euler is true, the array is
            size 1 x 6, if not, the orientation is described by quaternions, 
            making the array size 1 x 7. 
    """
    t_mats = get_t_mats(thetas, dh_params)
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


def get_rotational_error(target_quat: np.ndarray, source_quat: np.ndarray):
    """calculates the distance in radians between the current orientation of the
    end effector and the desired orientation, acquiring rotational error for the
    minimization algorithm.

    Args:
        target_quat (np.ndarray): a quaternion in the format [x, y, z, w]
            describing the target orientation of the end effector
        source_quat (np.ndarray): a quaternion in the format [x, y, z, w]
            describing the current orientation of the end effector 

    Returns:
        float: the distance in radians between the current orientation of the
            end effector and its desired orientation. Always positive. 
    """
    # clip rounds the input to arccos if it is larger in magnitude than 1
    # because any input greater in magnitude than 1 is invalid to arccos
    return np.arccos(np.clip(2*(np.dot(target_quat, source_quat)**2) - 1, -1, 1))


def get_error_vector(desired_end_effector_pose: np.ndarray,
                     current_end_effector_pose: np.ndarray):
    """calculates the error between the desired position and orientation of the
    end effector and its current position and orientation.

    Args:
        desired_end_effector_pose (np.ndarray): a 1 x 7 array describing the
            desired position and orientation of the end effector in the format
            [x position, y position, z position, x, y, z, w] in which the first
            three values describe the desired position of the end effector and
            the last four values describe its orientation as a quaternion. 
        current_end_effector_pose (np.ndarray): in the same format as 
            desired_end_effector_pose, but describes the current position and
            orientation of the end effector rather than the desired one. 

    Returns:
        np.ndarray: a 1 x 4 array describing the difference in x, y, z position
            between the desired end effector pose and the current end effector
            pose and the difference in radians between the desired orientation
            and the current orientation. 
    """
    error_vector = np.zeros(4)
    error_vector[:3] = np.subtract(
        desired_end_effector_pose[:3], current_end_effector_pose[:3])
    rotational_error = get_rotational_error(
        desired_end_effector_pose[3:], current_end_effector_pose[3:])
    error_vector[3] = rotational_error
    return error_vector


def get_mean_squared_error(error_vector: np.ndarray):
    """Converts the error vector into a single value to conduct optimization
    on and minimize because the optimization can only minimize a single value 
    rather than an entire vector of errors. 

    Args:
        error_vector (np.ndarray): a 1 x 4 error vector describing the
            difference in position and orientation between the current end
            effector pose and the desired pose. 

    Returns:
        float: the mean squared error between the current end effector pose and
            the desired pose. 
    """
    error_vector = error_vector**2
    error = np.sum(error_vector)
    error = error*(1/len(error_vector))
    return error


def objective_function(thetas: np.ndarray,
                       desired_end_effector_pose: np.ndarray,
                       dh_params: np.ndarray):
    """The objective function that the optimization algorithm seeks to minimize
    to conduct inverse kinematics. Returns the error between the desired pose of
    the end effector and the end effector pose produced by the inputted thetas,
    or joint angles.

    Args:
        thetas (np.ndarray): a 1 x n array of joint angles in the format
            [joint angle 1, joint angle 2, ... , joint angle n] in which n is 
            the number of joints on the robot arm. 
        desired_end_effector_pose (np.ndarray): a 1 x 7 array describing the
            desired position and orientation of the end effector in the format
            [x position, y position, z position, x, y, z, w] in which the first
            three values describe the desired position of the end effector and
            the last four values describe its orientation as a quaternion. 
        dh_params (np.ndarray): a 4 x n array containing the dh parameters to 
            describe every joint on the robot arm in the format
            [[theta offset 1, theta offset 2, ... , theta offset n],
            [alpha 1, alpha 2, ... , alpha n],
            [a 1, a 2, ... , a n], 
            [d 1, d 2, ... , d n]]
            where n is the number of joints on the robot arm. 

    Returns:
        float: error between the end effector pose produced from thetas and the
            desired end effector pose.
    """
    current_end_effector_pose = get_end_effector_pose(
        thetas, dh_params)
    error_vector = get_error_vector(
        desired_end_effector_pose, current_end_effector_pose)
    return get_mean_squared_error(error_vector)


def find_joint_angles(thetas_init: np.ndarray,
                      desired_end_effector_pose: np.ndarray,
                      dh_params: np.ndarray, smart_seed: bool = False,
                      tolerance: float = 1e-06, seeding_max_iters: int = 100):
    """Conducts inverse kinematics to find joint angles that result in the
    desired_end_effector_pose.

    Args:
        thetas_init (np.ndarray): a 1 x n array containing joint angles to seed
            the optimization algorithm with in the format
            [joint 1, joint 2, ... , joint n], where n is the number of joints
            in the robot arm. The closer these angles are to the true joint
            angles, the more likely it is that the optimization will find the
            true joint angles that lead to the desired end effector pose. 
        desired_end_effector_pose (np.ndarray): a 1 x 7 array describing the
            desired position and orientation of the end effector in the format
            [x position, y position, z position, x, y, z, w] in which the first
            three values describe the desired position of the end effector and
            the last four values describe its orientation as a quaternion. 
        dh_params (np.ndarray): a 4 x n array containing the dh parameters to 
            describe every joint on the robot arm in the format
            [[theta offset 1, theta offset 2, ... , theta offset n],
            [alpha 1, alpha 2, ... , alpha n],
            [a 1, a 2, ... , a n], 
            [d 1, d 2, ... , d n]]
            where n is the number of joints on the robot arm.
        smart_seed (bool, optional): whether the algorithm should randomly seed
            the optimization until an error lower than the tolerance is achieved
            if the error is too high. Defaults to False.
        tolerance (float, optional): The error tolerated between the desired
            end effector pose and the end effector pose calculated from the
            joint values the optimization solved for. Defaults to 1e-06.
        seeding_max_iters (int, optional): the number of times to optimize again
            with a new random set of thetas_init before giving up on achieving
            the error tolerance. Defaults to 100.

    Returns:
        Tuple(np.ndarray, bool): the set of joint angles the inverse kinematics
            algorithm solved for and whether the joint angles achieve the 
            desired end effector pose within the error tolerance in the format
            (joint angles, whether algorithm achieved error within tolerance)
    """
    solved = False
    result = minimize(objective_function, thetas_init,
                      (desired_end_effector_pose, dh_params,), method='BFGS')
    error = result.fun
    if error <= tolerance:
        solved = True
    joint_angles = result.x

    # trying new random seeds until error is within tolerance
    if not solved:
        if smart_seed:
            num_joints = len(dh_params[0])
            for i in range(seeding_max_iters):
                new_seed = np.array([random.uniform(-math.pi, math.pi)
                                    for j in range(num_joints)])
                new_result = minimize(objective_function, new_seed,
                                      (desired_end_effector_pose, dh_params,),
                                      method='BFGS')
                new_joint_angles = new_result.x
                new_error = new_result.fun
                if new_error < error:
                    error = new_error
                    joint_angles = new_joint_angles
                if new_error <= tolerance:
                    solved = True
                    break
    return (joint_angles, solved)

# ************************************** #
#     HELPER FUNCTIONS FOR ANIMATIONS    #
# ************************************** #


def draw_coordinate_system(plotter: pv.Plotter, t_mat: np.ndarray,
                           base: bool = False, name: str = ""):
    """draws a coordinate system on the plotter with the orientation and
    position described by t_mat and with the x, y, and z axes denoted by red,
    green, and blue colors. 

    Args:
        plotter (pv.Plotter): the plotter object to draw the coordinate system
            on.
        t_mat (np.ndarray): a transformation matrix describing the position and
            orientation of the coordinate system relative to the base reference
            frame. 
        base (bool, optional): Whether the coordinate system being drawn
            represents the base of the robot. Defaults to False.
        name (str, optional): the name of the joint whose coordinate system is
            being drawn so that the position and orientation of the coordinate
            system can be updated with user input. Rather than a new coordinate
            system being drawn, coordinate systems with names can be updated.
            Defaults to "".
    """
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


def draw_links(plotter: pv.Plotter, joint_positions: np.ndarray):
    """Draws the backbone of the robot arm on plotter given a set of joint
    positions relative to the base of the arm. 

    Args:
        plotter (pv.Plotter): the pyvista plotter object to draw the robot arm
            on
        joint_positions (np.ndarray): an n x 3 array denoting the position of 
            each joint relative to the base of the robot in the format 
            [[x 1, y 1, z 1], [x 2, y 2, z 2], ... , [x n, y n, z n]] where
            n = number of joints + 1.
    """
    poly = pv.PolyData()
    poly.points = joint_positions
    cells = np.full((len(joint_positions)-1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(joint_positions)-1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(joint_positions), dtype=np.int_)
    poly.lines = cells
    poly["scalars"] = np.arange(poly.n_points)
    tube = poly.tube(radius=10)
    plotter.add_mesh(tube, color="black", name="robot skeleton", opacity=0.5)


def draw_robot_arm(thetas: np.ndarray, p: pv.Plotter, dh_params: np.ndarray):
    """Draws the robot arm on the plotter, with a coordinate system representing
    each joint and a robot backbone connecting each joint. 

    Args:
        thetas (np.ndarray): a 1 x n array with joint angles in the format
            [joint angle 1, joint angle 2, ... , joint angle n], where n is the
            number of joints. 
        p (pv.Plotter): the plotter object to draw the robot arm on.
        dh_params (np.ndarray): a 4 x n array containing the dh parameters to 
            describe every joint on the robot arm in the format
            [[theta offset 1, theta offset 2, ... , theta offset n],
            [alpha 1, alpha 2, ... , alpha n],
            [a 1, a 2, ... , a n], 
            [d 1, d 2, ... , d n]]
            where n is the number of joints on the robot arm.
    """
    num_joints = len(thetas)
    joint_positions = np.zeros((num_joints + 1, 3))
    t_mats = get_t_mats(thetas, dh_params)
    for i, t_mat in enumerate(t_mats):
        joint_positions[i + 1, :3] = t_mat[:3, 3]
        draw_coordinate_system(p, t_mat, name=f"theta{i}")
    draw_links(p, joint_positions)


def convert_pose_to_quat(pose_in_euler: np.ndarray):
    """converts end effector pose in euler angles to pose with orientation
    described as a quaternion. 

    Args:
        pose_in_euler (np.ndarray): a 1 x 6 array describing the position and
            orientation of the end effector in the format [x, y, z, Z, Y, X] 
            with orientation represented by euler angles following the ZYX
            convention denoted by capital letters. 

    Returns:
        np.ndarray: a 1 x 7 array describing the position and orientation of the 
            end effector in the format [x, y, z, X, Y, Z, W] with orientation
            represented by a quaternion denoted by capital letters. 
    """
    rot = Rotation.from_euler(
        'ZYX', pose_in_euler[3:], degrees=False)
    quat = rot.as_quat()
    pose_in_quat = np.zeros(7)
    pose_in_quat[:3] = pose_in_euler[:3]
    pose_in_quat[3:] = quat
    return pose_in_quat


def convert_pose_to_euler(pose_in_quat: np.ndarray):
    """converts end effector pose with orientation described in quaternions to 
    pose with orientation described in euler angles. 

    Args:
        pose_in_quat (np.ndarray): a 1 x 7 array describing the position and 
            orientation of the end effector in the format [x, y, z, X, Y, Z, W] 
            with orientation represented by a quaternion denoted by capital
            letters.

    Returns:
        [type]: a 1 x 6 array describing the position and orientation of the
            end effector in the format [x, y, z, Z, Y, X] with orientation
            represented by euler angles following the ZYX convention denoted by
            capital letters. 
    """
    rot = Rotation.from_quat(pose_in_quat[3:])
    euler = rot.as_euler("ZYX")
    pose_in_euler = np.zeros(6)
    pose_in_euler[:3] = pose_in_quat[:3]
    pose_in_euler[3:] = euler
    return pose_in_euler


def animate_forward_kinematics(dh_params: np.ndarray):
    """generates a GUI that allows users to manipulate the joint angles of the
    robot arm described by dh_params and watch the end effector move accordingly
    in an animation, demonstrating forward kinematics.

    Args:
        dh_params (np.ndarray): a 4 x n array containing the dh parameters to 
            describe every joint on the robot arm in the format
            [[theta offset 1, theta offset 2, ... , theta offset n],
            [alpha 1, alpha 2, ... , alpha n],
            [a 1, a 2, ... , a n], 
            [d 1, d 2, ... , d n]]
            where n is the number of joints on the robot arm.
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    num_joints = len(dh_params[0])
    p = pv.Plotter()
    draw_coordinate_system(p, np.eye(4), True)
    thetas = np.zeros(6)

    def callback(idx: int, theta: float):
        thetas[idx] = theta
        draw_robot_arm(thetas, p, dh_params)

    for i in range(num_joints):
        p.add_slider_widget(partial(callback, (i,)),
                            [-np.pi, np.pi], pointa=(0.7, 0.9-0.15*i),
                            pointb=(0.95, 0.9-0.15*i),
                            title=f"Theta {i}", event_type="end")
    p.show()


def animate_inverse_kinematics_sphere(dh_params: np.ndarray):
    """generates a GUI that allows users to manipulate the position and
    orientation of the end effector of the robot arm described by dh_params
    and watch the joint angles move accordingly in an animation, demonstrating
    inverse kinematics. The end effector's position can be changed by clicking
    and dragging a 3D sphere around for the end effector to follow. 

    Args:
        dh_params (np.ndarray): a 4 x n array containing the dh parameters to 
            describe every joint on the robot arm in the format
            [[theta offset 1, theta offset 2, ... , theta offset n],
            [alpha 1, alpha 2, ... , alpha n],
            [a 1, a 2, ... , a n], 
            [d 1, d 2, ... , d n]]
            where n is the number of joints on the robot arm.
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    p = pv.Plotter()
    draw_coordinate_system(p, np.eye(4), base=True)
    thetas_init = np.zeros(6)
    desired_end_effector_pose = get_end_effector_pose(
        thetas_init, dh_params, euler=True)

    def callback(idx: int, pose_param: float):
        desired_end_effector_pose[idx] = pose_param
        desired_end_effector_pose_in_quat = convert_pose_to_quat(
            desired_end_effector_pose)
        thetas, solved = find_joint_angles(
            thetas_init, desired_end_effector_pose_in_quat, dh_params)
        if solved:
            thetas_init[:] = thetas

        draw_robot_arm(thetas, p, dh_params)

    p.add_sphere_widget(
        partial(callback, range(0, 3)),
        center=desired_end_effector_pose[:3], radius=10,
        color="#00FF00", test_callback=False)

    end_effector_pose_labels = ["x_rot", "y_rot", "z_rot"]
    for i in range(3):
        p.add_slider_widget(partial(callback, i + 3),
                            [-np.pi, np.pi],
                            value=desired_end_effector_pose[i + 3],
                            pointa=(0.7, 0.9-0.15*i), pointb=(0.95, 0.9-0.15*i),
                            title=end_effector_pose_labels[i], event_type="end")
    p.show()


def animate_inverse_kinematics_sliders(dh_params: np.ndarray,
                                       position_bound: float = 700):
    """generates a GUI that allows users to manipulate the position and
    orientation of the end effector of the robot arm described by dh_params
    and watch the joint angles move accordingly in an animation, demonstrating
    inverse kinematics. The end effector's position can be changed by dragging
    the slider for each axis.

    Args:
        dh_params (np.ndarray): a 4 x n array containing the dh parameters to 
            describe every joint on the robot arm in the format
            [[theta offset 1, theta offset 2, ... , theta offset n],
            [alpha 1, alpha 2, ... , alpha n],
            [a 1, a 2, ... , a n], 
            [d 1, d 2, ... , d n]]
            where n is the number of joints on the robot arm.
        position_bound (float, optional): because sliders are being used to
            indicate a desired position, bounds need to be placed on the x, y,
            and z values. Defaults to 700.
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    p = pv.Plotter()
    draw_coordinate_system(p, np.eye(4), base=True)
    thetas_init = np.zeros(6)
    desired_end_effector_pose = get_end_effector_pose(
        thetas_init, dh_params, euler=True)

    def callback(idx, pose_param):
        desired_end_effector_pose[idx] = pose_param
        desired_end_effector_pose_quat = convert_pose_to_quat(
            desired_end_effector_pose)
        thetas, solved = find_joint_angles(
            thetas_init, desired_end_effector_pose_quat, dh_params)
        if solved:
            thetas_init[:] = thetas
        draw_robot_arm(thetas, p, dh_params)
    end_effector_pose_labels = ["x", "y", "z", "x_rot", "y_rot", "z_rot"]
    for i in range(6):
        if i <= 2:
            lower_bound = -position_bound
            upper_bound = -lower_bound
        else:
            lower_bound = -np.pi
            upper_bound = -lower_bound
        p.add_slider_widget(partial(callback, (i,)),
                            [lower_bound, upper_bound],
                            value=desired_end_effector_pose[i],
                            pointa=(0.7, 0.9-0.15*i),
                            pointb=(0.95, 0.9-0.15*i),
                            title=end_effector_pose_labels[i], event_type="end")

    p.show()

# ******************************* #
#     MORE HELPFUL FUNCTIONS      #
# ******************************* #


def wrap_joint_angles(joint_angles: np.ndarray, radians: bool = True):
    """Confines a set of joint angles to the period -pi, pi.

    Args:
        joint_angles (np.ndarray): a 1 x n array of joint angles in radians or
            degrees.
        radians (bool, optional): Whether the input is in radians.
            Defaults to True.

    Returns:
        np.ndarray: a 1 x n array of the inputted joint angles confined to the
            period -pi, pi
    """
    if radians:
        period = np.pi
    else:
        period = 180
    return (joint_angles + period) % (2 * period) - period


# ************* #
#     MAIN      #
# ************* #

if __name__ == "__main__":
    ar3_alpha_vals = [-(math.pi/2), 0, math.pi/2, -(math.pi/2), math.pi/2, 0]
    ar3_a_vals = [64.2, 305, 0, 0, 0, 0]
    ar3_d_vals = [169.77, 0, 0, -222.63, 0, -36.25]
    ar3_theta_offsets = [0, 0, -math.pi/2, 0, 0, math.pi]

    ar3_dh_params = np.array(
        [ar3_theta_offsets, ar3_alpha_vals, ar3_a_vals, ar3_d_vals])

    animate_forward_kinematics(ar3_dh_params)
