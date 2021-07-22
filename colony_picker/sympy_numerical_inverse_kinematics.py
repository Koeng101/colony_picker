import numpy as np
import sympy as sp

# Denavit-Hartenberg Parameters of AR3 provided by
# AR2 Version 2.0 software executable files from
# https://www.anninrobotics.com/downloads
# parameters are the same between the AR2 and AR3
alphas = [-(sp.pi/2), 0, sp.pi/2, -(sp.pi/2), sp.pi/2, 0]
a_vals = [64.2, 305, 0, 0, 0, 0]
d_vals = [169.77, 0, 0, -222.63, 0, -36.25]
theta_offsets = [0, -(sp.pi/2), 0, 0, 0, sp.pi]

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
    t_mats = []
    for joint_idx in range(0, NUM_JOINTS):
        t_mat = get_t_mat(thetas[joint_idx],
                          theta_offsets[joint_idx],
                          a_vals[joint_idx],
                          alphas[joint_idx],
                          d_vals[joint_idx])
        end_effector_t_mat = end_effector_t_mat * t_mat
        t_mats.append(end_effector_t_mat)
    return (end_effector_t_mat, thetas, t_mats)


def get_jacobian():

    return 0


def get_jacobian(theta_vals):
    unevaluated_t_mat, theta_names, t_mats = get_end_effector_t_mat()
    end_effector_pos = unevaluated_t_mat[:3, 3]
    jacobian = np.zeros((6, NUM_JOINTS))
    theta_substitutions = []
    for joint_idx in range(NUM_JOINTS):
        theta_substitutions.append(
            (theta_names[joint_idx], theta_vals[joint_idx]))
        t_mat = t_mats[joint_idx]
        rot_axis = t_mat[:3, 2]
        jacobian[3:, joint_idx] = rot_axis
    print()
    print("SYMPY END EFFECTOR POS:")
    print(end_effector_pos.subs(theta_substitutions))
    print()
    for axis_idx in range(3):
        end_effector_component_equation = end_effector_pos[axis_idx]
        for joint_idx in range(NUM_JOINTS):
            derivative = sp.diff(
                end_effector_component_equation, theta_names[joint_idx])
            jacobian[axis_idx, joint_idx] = derivative.subs(
                theta_substitutions)
    print("SYMPY JACOBIAN TRANSLATIONAL COMPONENTS: ")
    print(jacobian)
    return jacobian


def get_euler_from_rot_mat(rot_mat):
    return np.array([sp.atan2(rot_mat[2, 1], rot_mat[2, 2]),
                     sp.atan2(-rot_mat[2, 0], sp.sqrt(np.square(rot_mat[2, 1])
                                                      + np.square(rot_mat[2, 2]))),
                     sp.atan2(rot_mat[1, 0], rot_mat[0, 0])])


def get_end_effector_pose(thetas):
    end_effector_t_mat = get_end_effector_t_mat(thetas)
    end_effector_position = end_effector_t_mat[:3, 3]
    rot_mat = end_effector_t_mat[:3, :3]
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


thetas = np.zeros(NUM_JOINTS)
get_translational_jacobian_components(thetas)
