"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

import numpy as np
# expm is a matrix exponential function
from scipy.linalg import expm


def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def FK_dh(dh_params, joint_angles, link):
    """!
    @brief      Get the 4x4 transformation matrix from link to world

                TODO: implement this function

                Calculate forward kinematics for rexarm using DH convention

                return a transformation matrix representing the pose of the desired link

                note: phi is the euler angle about the y-axis in the base frame

    @param      dh_params     The dh parameters as a 2D list each row represents a link and has the format [a, alpha, d,
                              theta]
    @param      joint_angles  The joint angles of the links
    @param      link          The link to transform from

    @return     a transformation matrix representing the pose of the desired link
    """
    T = np.eye(4)
    for i in range(link + 1):
        a, alpha, d, theta = dh_params[i]
        theta += joint_angles[i]
        T_link = get_transform_from_dh(a, alpha, d, theta)
        T = np.dot(T, T_link)
    return T


def get_transform_from_dh(a, alpha, d, theta):
    """!
    @brief      Gets the transformation matrix T from dh parameters.

    TODO: Find the T matrix from a row of a DH table

    @param      a      a meters
    @param      alpha  alpha radians
    @param      d      d meters
    @param      theta  theta radians

    @return     The 4x4 transformation matrix.
    """
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])


def get_euler_angles_from_T(T):
    """!
    @brief      Gets the euler angles from a transformation matrix.

                TODO: Implement this function return the 3 Euler angles from a 4x4 transformation matrix T
                If you like, add an argument to specify the Euler angles used (xyx, zyz, etc.)

    @param      T     transformation matrix

    @return     The euler angles from T.
    """
    if T[2, 0] != 1 and T[2, 0] != -1:
        pitch = -np.arcsin(T[2, 0])
        roll = np.arctan2(T[2, 1] / np.cos(pitch), T[2, 2] / np.cos(pitch))
        yaw = np.arctan2(T[1, 0] / np.cos(pitch), T[0, 0] / np.cos(pitch))
    else:
        yaw = 0
        if T[2, 0] == -1:
            pitch = np.pi / 2
            roll = np.arctan2(T[0, 1], T[0, 2])
        else:
            pitch = -np.pi / 2
            roll = np.arctan2(-T[0, 1], -T[0, 2])
    return roll, pitch, yaw


def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the 6DOF pose vector from a 4x4 transformation matrix T

    @param      T     transformation matrix

    @return     The pose vector from T.
    """
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    roll, pitch, yaw = get_euler_angles_from_T(T)
    return [x, y, z, roll, pitch, yaw]


def FK_pox(joint_angles, m_mat, s_lst):
    """!
    @brief      Get a  representing the pose of the desired link

                TODO: implement this function, Calculate forward kinematics for rexarm using product of exponential
                formulation return a 4x4 homogeneous matrix representing the pose of the desired link

    @param      joint_angles  The joint angles
                m_mat         The M matrix
                s_lst         List of screw vectors

    @return     a 4x4 homogeneous matrix representing the pose of the desired link
    """
    T = np.eye(4)
    for i in range(len(joint_angles)):
        # print(i)
        # print("line 1")
        # s_matrix = s_lst[i]
        s_matrix = to_s_matrix(s_lst[i,0:3], s_lst[i,3:6])
        # print(s_matrix)
        # print("line 2")
        theta = joint_angles[i]
        # print(theta)
        # print("line 3")
        T = T @ expm(s_matrix * theta)
        # print("line 4")
    

    T = T @ m_mat
    return T


def to_s_matrix(w, v):
    """!
    @brief      Convert to s matrix.

    TODO: implement this function
    Find the [s] matrix for the POX method e^([s]*theta)

    @param      w     { parameter_description }
    @param      v     { parameter_description }

    @return     { description_of_the_return_value }
    """
    w_x, w_y, w_z = w
    v_x, v_y, v_z = v

    s_matrix = np.array([
        [0, -w_z, w_y, v_x],
        [w_z, 0, -w_x, v_y],
        [-w_y, w_x, 0, v_z],
        [0, 0, 0, 0]
    ])


def IK_geometric(dh_params, pose):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose vector as np.array to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose vector as np.array 

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    pass



import numpy as np

# Define joint limits (in radians)
joint_limits = {
    "theta1": (-np.pi, np.pi),                # Waist
    "theta2": (np.radians(-108), np.radians(113)),  # Shoulder
    "theta3": (np.radians(-108), np.radians(93)),   # Elbow
    "theta4": (np.radians(-100), np.radians(123)),  # Wrist Angle
    "theta5": (-np.pi, np.pi)                 # Wrist Rotate (ignored for now)
}

def check_joint_limits(theta, joint_name):
    """Check if a joint angle is within its limits."""
    min_limit, max_limit = joint_limits[joint_name]
    return min_limit <= theta <= max_limit

def select_within_limits(solutions):
    """Select the first solution within joint limits."""
    for solution in solutions:
        if all(check_joint_limits(angle, joint) for angle, joint in zip(solution, joint_limits.keys())):
            return solution
    raise ValueError("No solutions found within joint limits")

def IK_geometric(pose, block_ori=None, isVertical_Pick=False, force_t5 = None):
    """
    Calculate inverse kinematics for the ReactorX-200 arm.
    
    Args:
        pose: Desired end-effector pose as a numpy array [x, y, z, phi].
        block_ori: Orientation of the block (used for wrist alignment).
        isVertical_Pick: Boolean indicating if the wrist should be vertically aligned.
    
    Returns:
        Tuple indicating success (boolean) and joint angles [theta1, theta2, theta3, theta4, theta5].
    """
    # Link lengths
    l1 = 103.91
    l2 = 200.
    l3 = 50.
    l4 = 205.73
    l5 = 200.
    l6 = 174.15

    alpha = np.arctan(l3 / l2)
    x, y, z, phi = pose
    coord = np.array([[x], [y], [z]])
    
    # x += 19
    # y += 7

    if z < -2:
        print("[Target Pose Error] Manipulator cannot reach below the base plane.")
        return False, [0, 0, 0, 0, 0]
    
    # Calculate base joint angle theta1
    theta1 = np.arctan2(-x, y)
    
    # Calculate position of the wrist center
    lastlink_unit = np.array([-np.sin(theta1) * np.cos(phi), np.cos(theta1) * np.cos(phi), -np.sin(phi)])
    xc, yc, zc = np.array([x, y, z])- l6 * lastlink_unit

    # # Check if the position is within reach
    if np.sqrt(xc**2 + yc**2 + (zc - l1)**2) > (l4 + l5):
        print("[IK Error] Unreachable Position: Target out of workspace.")
        return False, [0, 0, 0, 0, 0]

    r = np.sqrt(xc**2 + yc**2)
    s = zc - l1
    beta = np.arccos((l4**2 + l5**2 - r**2 - s**2) / (2.0 * l4 * l5))
    theta3 = np.pi/2 + alpha - beta
    theta2 = (np.pi/2 - alpha - np.arccos(r / np.sqrt(r**2 + s**2)) 
              - np.arccos((l4**2 + r**2 + s**2 - l5**2) / (2 * l4 * np.sqrt(r**2 + s**2))))
    if theta1 < 0:
        theta1+= (-0.01)
    else:
        pass
    # theta2 += (-0.03)
    theta3 += (-0.065)

    theta4 = phi - theta2 - theta3 - 0.08

    # Joint limits check
    joint_limits = [
        (-np.pi, np.pi),           # Waist
        (-108 * np.pi / 180, 113 * np.pi / 180),  # Shoulder
        (-108 * np.pi / 180, 93 * np.pi / 180),   # Elbow
        (-100 * np.pi / 180, 123 * np.pi / 180),  # Wrist Angle
        (-np.pi, np.pi)            # Wrist Rotate
    ]

    if force_t5 is not None:
        theta5 = force_t5
    else:
        theta5 = 0
        
    # Check if each calculated joint angle is within limits
    thetas = [theta1, theta2, theta3, theta4, theta5]
    for i, (theta, limits) in enumerate(zip(thetas, joint_limits)):
        if not (limits[0] <= theta <= limits[1]):
            print(f"[Joint Limit Error] Joint {i+1} angle {theta} out of limits {limits}.")
            return False, [0, 0, 0, 0, 0]

    return True, thetas
