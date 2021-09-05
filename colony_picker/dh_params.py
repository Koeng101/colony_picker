"""Stores Denavit-Hartenberg parameters for different robot arms. 
"""
import math
import numpy as np

# *************************************** #
#     AR3 6-AXIS ROBOT ARM DH PARAMS      #
# *************************************** #

ar3_alpha_vals = [-(math.pi/2), 0, math.pi/2, -(math.pi/2), math.pi/2, 0]
ar3_a_vals = [64.2, 305, 0, 0, 0, 0]
ar3_d_vals = [169.77, 0, 0, -222.63, 0, -36.25]
ar3_theta_offsets = [0, 0, -math.pi/2, 0, 0, math.pi]

AR3_DH_PARAMS = np.array(
    [ar3_theta_offsets, ar3_alpha_vals, ar3_a_vals, ar3_d_vals])
AR3_NUM_JOINTS = len(AR3_DH_PARAMS[0])

# *************************************** #
