from scipy.interpolate import CubicSpline
import numpy as np

def get_angle(v):
    """
    Calculate the angles (in radians) between a 3D vector and the principal axes (x, y, and z).

    Args:
    v (numpy.ndarray): A 3D vector.

    Returns:
    tuple: A tuple containing the angles (thetax, thetay, thetaz) with respect to the x, y, and z axes.
    """
    axis_x = np.array([1,0,0])
    axis_y = np.array([0,1,0])
    axis_z = np.array([0,0,1])

    thetax = axis_x.dot(v)/(np.linalg.norm(axis_x) * np.linalg.norm(v))
    thetay = axis_y.dot(v)/(np.linalg.norm(axis_y) * np.linalg.norm(v))
    thetaz = axis_z.dot(v)/(np.linalg.norm(axis_z) * np.linalg.norm(v))

    return thetax, thetay, thetaz

def get_position(v, angles):
    """
    Calculate the position in 3D space given a 3D vector and a tuple of angles (in radians).

    Args:
    v (numpy.ndarray): A 3D vector representing distance.
    angles (tuple): A tuple of angles (thetax, thetay, thetaz) in radians.

    Returns:
    tuple: A tuple containing the 3D position (x, y, z) in space.

    Note: The position is calculated relative to the origin based on the input vector and angles.
    """
    r = np.linalg.norm(v)
    x = r*angles[0]
    y = r*angles[1]
    z = r*angles[2]
    
    return  x,y,z

def calculate_angle(fullbody, jointConnect):
    """
    Calculate angles between specified joints in a full-body model represented by a sequence of 3D vectors.

    Args:
    fullbody (numpy.ndarray): An array of 3D vectors representing the full-body model.
    jointConnect (list): A list of joint index pairs specifying the joints for angle calculation.

    Returns:
    numpy.ndarray: An array (AngleList) with calculated angles between specified joints for each frame.

    Note: The input represents the full-body model, and the output provides joint angles between specified joints based on jointConnect.
    """
    AngleList = np.zeros_like(fullbody)
    for i, frame in enumerate(fullbody):
        for joint in jointConnect:
            v = frame[joint[0]:joint[0]+3] - frame[joint[1]:joint[1]+3]
            AngleList[i][joint[0]:joint[0]+3] = list(get_angle(v))
    return AngleList


def calculate_position(fullbody, TP, jointConnect):
    """
    Calculate joint positions in a full-body model relative to a reference pose.

    Args:
    fullbody (numpy.ndarray): An array of 3D vectors representing the full-body model.
    TP (numpy.ndarray): A reference pose used as the basis for position calculation.
    jointConnect (list): A list of joint index pairs specifying the joints for position calculation.

    Returns:
    numpy.ndarray: An array (PosList) containing the calculated joint positions relative to the reference pose.

    Note: This function calculates joint positions in the model relative to a reference pose (TP).
    """
    PosList = np.zeros_like(fullbody)
    for i, frame in enumerate(fullbody):
        for joint in jointConnect:
            v = TP[joint[0]:joint[0]+3] - TP[joint[1]:joint[1]+3]
            angles = frame[joint[0]:joint[0]+3]
            root = PosList[i][joint[1]:joint[1]+3]
            PosList[i][joint[0]:joint[0]+3] = np.array(list(get_position(v, angles)))+root

    return PosList

def normalize(data, joints):
    """
    Normalize motion data by centering it around the pelvis joint.

    Args:
    data (numpy.ndarray): Motion data represented as a 3D array of joint positions.
    joints (dict): A dictionary mapping joint names to their indices in the data.

    Returns:
    numpy.ndarray: Normalized motion data with respect to the pelvis joint.

    Note: This function reshapes the input data, centers it around the pelvis joint, and returns the normalized motion data.
    """
    data = data.reshape(data.shape[0], int(data.shape[1]/3), 3)
    normal_data = []
    for i, frame in enumerate(data):
        root = (frame[joints['RightThigh']]+frame[joints['LeftThigh']])/2
        data[i, joints['Pelvis']] = root
        normal_data.append([])
        for node in frame:
            normal_data[-1].extend(node - root)
    return np.array(normal_data)

def crossfading(motion1, motion2, cross_frame = 5):
    """
    Perform crossfading between two motion sequences to create a smooth transition.

    Args:
    motion1 (numpy.ndarray): The first motion sequence.
    motion2 (numpy.ndarray): The second motion sequence.
    cross_frame (int): The number of frames for the crossfade transition.

    Returns:
    numpy.ndarray: The final motion sequence after crossfading.

    Note: This function blends motion1 and motion2 together with a crossfade transition
    over the specified number of frames (cross_frame) to create a smooth transition.
    """
    motion1 = motion1[:-cross_frame, :]
    motion2 = motion2[cross_frame:, :]
    last_data_motion1 = motion1[-1, :]
    first_data_motion2 = motion2[0, :]

    # Define the number of interpolation steps
    num_steps = cross_frame * 2

    # Create an array of interpolation points
    alpha_values = np.linspace(0, 1, num_steps + 2)[1:-1]  # Exclude 0 and 1

    # Create a cubic spline interpolation function for each dimension
    interpolated_data = np.zeros((num_steps, motion1.shape[1]))
    for dim in range(motion1.shape[1]):
        cs = CubicSpline([0, 1], [last_data_motion1[dim], first_data_motion2[dim]])
        interpolated_data[:, dim] = cs(alpha_values)

    # Concatenate motion1, interpolated_data, and motion2 to create the final data
    final_data = np.vstack((motion1, interpolated_data, motion2))
    
    return final_data