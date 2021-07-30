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

def get_euler_from_rot_mat(rot_mat):
    from scipy.spatial.transform import Rotation
    r = Rotation.from_matrix(rot_mat)
    return r.as_euler("xyz")
    all_tait_bryan = [
        "xyz",
        "xzy",
        "xyx",
        "xzx",
        "yzx",
        "yxz",
        "yzy",
        "yxy",
        "zyx",
        "zxy",
        "zyz",
        "zxz",
        "xyz".upper(),
        "xzy".upper(),
        "xyx".upper(),
        "xzx".upper(),
        "yzx".upper(),
        "yxz".upper(),
        "yzy".upper(),
        "yxy".upper(),
        "zyx".upper(),
        "zxy".upper(),
        "zyz".upper(),
        "zxz".upper()
    ]
    expected_euler = [-1.395457519, 1.745114494, -1.582076348]
    for tb in all_tait_bryan:
        print("TAIT BRYAN EULER: ")
        print(r.as_euler(tb))
        print()
        # print(np.array_equal(np.sort(r.as_euler(tb)), np.sort(expected_euler)))

        # return r.as_euler("ZXZ")

    # excel spreadhseet implementation
    # r_y = np.arctan2(np.sqrt(rot_mat[0, 2]**2 + rot_mat[1, 2]**2), -rot_mat[2, 2])
    # r_x = np.arctan2(rot_mat[2, 0]/r_y, rot_mat[2, 1]/r_y)
    # r_z = np.arctan2(rot_mat[0, 2]/r_y, rot_mat[1, 2]/r_y)
    # final_euler = [r_x, r_y, r_z]

    # excel spreadsheet formula (differs from implementation and might be the source of a bug)
    # I should check if this formula matches an euler angle convention
    r_y = np.arctan2(-rot_mat[2, 0],
                     np.sqrt(rot_mat[0, 0]**2 + rot_mat[1, 0]**2))
    r_x = np.arctan2(rot_mat[2, 1]/np.cos(r_y), rot_mat[2, 2]/np.cos(r_y))
    r_z = np.arctan2(rot_mat[1, 0]/np.cos(r_y), rot_mat[0, 0]/np.cos(r_y))
    final_euler = [r_x, r_y, r_z]
    print("SPREADSHEET EULER: ")
    print(final_euler)
    print()

    # if np.any(np.greater(np.abs(final_euler), math.pi)):
    #     print([r_x, r_y, r_z])
    # return np.array([r_x, r_y, r_z])
