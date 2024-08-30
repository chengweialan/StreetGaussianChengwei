
import os
import cv2
import sys
import open3d as o3d
import imageio.v2 as imageio
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from scene.scene_utils import CameraInfo, SceneInfo, getNerfppNorm, fetchPly, storePly
from pathlib import Path


_sem2label = {
    'Car': 0,
    'Van': 1,
    'Truck': 2,
}

_sem2model = {
    'Sedan4Door': 0,
    'Hatchback': 1,
    'Hybrid': 2,
    'SUV': 3,
    'Firetruck_small_eu': 4,
    'MCP2_Ambulance_A': 5,
    'Ford_F600_CargoVan': 6,
    'Renault_Kangoo': 7,
    'MCP2_BusA_01': 8,
}

_sem2color = {
    'Red': 0,
    'Silver': 1,
    'Blue': 2,
    'Black': 3,
    'White': 4,
    'Brown': 5,
    'Grey': 6,
}

camera_ls = [0, 1]

"""
Most function brought from MARS
https://github.com/OPEN-AIR-SUN/mars/blob/69b9bf9d992e6b9f4027dfdc2a741c2a33eef174/mars/data/mars_kitti_dataparser.py
"""

def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def transform_poses_pca(poses, fix_radius=0):
    """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
    
    From https://github.com/SuLvXiangXin/zipnerf-pytorch/blob/af86ea6340b9be6b90ea40f66c0c02484dfc7302/internal/camera_utils.py#L161
  """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    if fix_radius>0:
        scale_factor = 1./fix_radius
    else:
        scale_factor = 1. / (np.max(np.abs(poses_recentered[:, :3, 3])) + 1e-5)
        scale_factor = min(1 / 10, scale_factor)

    # To match the size
    # scale_factor=1
    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

    return poses_recentered, transform, scale_factor

def kitti_string_to_float(str):
    return float(str.split("e")[0]) * 10 ** int(str.split("e")[1])


def get_rotation(roll, pitch, heading):
    s_heading = np.sin(heading)
    c_heading = np.cos(heading)
    rot_z = np.array([[c_heading, -s_heading, 0], [s_heading, c_heading, 0], [0, 0, 1]])

    s_pitch = np.sin(pitch)
    c_pitch = np.cos(pitch)
    rot_y = np.array([[c_pitch, 0, s_pitch], [0, 1, 0], [-s_pitch, 0, c_pitch]])

    s_roll = np.sin(roll)
    c_roll = np.cos(roll)
    rot_x = np.array([[1, 0, 0], [0, c_roll, -s_roll], [0, s_roll, c_roll]])

    rot = np.matmul(rot_z, np.matmul(rot_y, rot_x))

    return rot

# deal with kitti_mot
def tracking_calib_from_txt(calibration_path):
    """
    Extract tracking calibration information from a KITTI tracking calibration file.

    This function reads a KITTI tracking calibration file and extracts the relevant
    calibration information, including projection matrices and transformation matrices
    for camera, LiDAR, and IMU coordinate systems.

    Args:
        calibration_path (str): Path to the KITTI tracking calibration file.

    Returns:
        dict: A dictionary containing the following calibration information:
            P0, P1, P2, P3 (np.array): 3x4 projection matrices for the cameras.
            Tr_cam2camrect (np.array): 4x4 transformation matrix from camera to rectified camera coordinates.
            Tr_velo2cam (np.array): 4x4 transformation matrix from LiDAR to camera coordinates.
            Tr_imu2velo (np.array): 4x4 transformation matrix from IMU to LiDAR coordinates.
    """
    # Read the calibration file
    f = open(calibration_path)
    calib_str = f.read().splitlines()

    # Process the calibration data
    calibs = []
    for calibration in calib_str:
        calibs.append(np.array([kitti_string_to_float(val) for val in calibration.split()[1:]]))

    # Extract the projection matrices
    P0 = np.reshape(calibs[0], [3, 4])
    P1 = np.reshape(calibs[1], [3, 4])
    P2 = np.reshape(calibs[2], [3, 4])
    P3 = np.reshape(calibs[3], [3, 4])

    # Extract the transformation matrix for camera to rectified camera coordinates
    Tr_cam2camrect = np.eye(4)
    R_rect = np.reshape(calibs[4], [3, 3])
    Tr_cam2camrect[:3, :3] = R_rect

    # Extract the transformation matrices for LiDAR to camera and IMU to LiDAR coordinates
    Tr_velo2cam = np.concatenate([np.reshape(calibs[5], [3, 4]), np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)
    Tr_imu2velo = np.concatenate([np.reshape(calibs[6], [3, 4]), np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)

    return {
        "P0": P0,
        "P1": P1,
        "P2": P2,
        "P3": P3,
        "Tr_cam2camrect": Tr_cam2camrect,
        "Tr_velo2cam": Tr_velo2cam,
        "Tr_imu2velo": Tr_imu2velo,
    }

# deal with kitti_mot
def calib_from_txt(calibration_path):
    """
    Read the calibration files and extract the required transformation matrices and focal length.

    Args:
        calibration_path (str): The path to the directory containing the calibration files.

    Returns:
        tuple: A tuple containing the following elements:
            traimu2v (np.array): 4x4 transformation matrix from IMU to Velodyne coordinates.
            v2c (np.array): 4x4 transformation matrix from Velodyne to left camera coordinates.
            c2leftRGB (np.array): 4x4 transformation matrix from left camera to rectified left camera coordinates.
            c2rightRGB (np.array): 4x4 transformation matrix from right camera to rectified right camera coordinates.
            focal (float): Focal length of the left camera.
    """
    c2c = []

    # Read and parse the camera-to-camera calibration file
    f = open(os.path.join(calibration_path, "calib_cam_to_cam.txt"), "r")
    cam_to_cam_str = f.read()
    [left_cam, right_cam] = cam_to_cam_str.split("S_02: ")[1].split("S_03: ")
    cam_to_cam_ls = [left_cam, right_cam]

    # Extract the transformation matrices for left and right cameras
    for i, cam_str in enumerate(cam_to_cam_ls):
        r_str, t_str = cam_str.split("R_0" + str(i + 2) + ": ")[1].split("\nT_0" + str(i + 2) + ": ")
        t_str = t_str.split("\n")[0]
        R = np.array([kitti_string_to_float(r) for r in r_str.split(" ")])
        R = np.reshape(R, [3, 3])
        t = np.array([kitti_string_to_float(t) for t in t_str.split(" ")])
        Tr = np.concatenate([np.concatenate([R, t[:, None]], axis=1), np.array([0.0, 0.0, 0.0, 1.0])[None, :]])

        t_str_rect, s_rect_part = cam_str.split("\nT_0" + str(i + 2) + ": ")[1].split("\nS_rect_0" + str(i + 2) + ": ")
        s_rect_str, r_rect_part = s_rect_part.split("\nR_rect_0" + str(i + 2) + ": ")
        r_rect_str = r_rect_part.split("\nP_rect_0" + str(i + 2) + ": ")[0]
        R_rect = np.array([kitti_string_to_float(r) for r in r_rect_str.split(" ")])
        R_rect = np.reshape(R_rect, [3, 3])
        t_rect = np.array([kitti_string_to_float(t) for t in t_str_rect.split(" ")])
        Tr_rect = np.concatenate(
            [np.concatenate([R_rect, t_rect[:, None]], axis=1), np.array([0.0, 0.0, 0.0, 1.0])[None, :]]
        )

        c2c.append(Tr_rect)

    c2leftRGB = c2c[0]
    c2rightRGB = c2c[1]

    # Read and parse the Velodyne-to-camera calibration file
    f = open(os.path.join(calibration_path, "calib_velo_to_cam.txt"), "r")
    velo_to_cam_str = f.read()
    r_str, t_str = velo_to_cam_str.split("R: ")[1].split("\nT: ")
    t_str = t_str.split("\n")[0]
    R = np.array([kitti_string_to_float(r) for r in r_str.split(" ")])
    R = np.reshape(R, [3, 3])
    t = np.array([kitti_string_to_float(r) for r in t_str.split(" ")])
    v2c = np.concatenate([np.concatenate([R, t[:, None]], axis=1), np.array([0.0, 0.0, 0.0, 1.0])[None, :]])

    # Read and parse the IMU-to-Velodyne calibration file
    f = open(os.path.join(calibration_path, "calib_imu_to_velo.txt"), "r")
    imu_to_velo_str = f.read()
    r_str, t_str = imu_to_velo_str.split("R: ")[1].split("\nT: ")
    R = np.array([kitti_string_to_float(r) for r in r_str.split(" ")])
    R = np.reshape(R, [3, 3])
    t = np.array([kitti_string_to_float(r) for r in t_str.split(" ")])
    imu2v = np.concatenate([np.concatenate([R, t[:, None]], axis=1), np.array([0.0, 0.0, 0.0, 1.0])[None, :]])

    # Extract the focal length of the left camera
    focal = kitti_string_to_float(left_cam.split("P_rect_02: ")[1].split()[0])

    return imu2v, v2c, c2leftRGB, c2rightRGB, focal

# deal with kitti_mot
def get_poses_calibration(basedir, oxts_path_tracking=None, selected_frames=None):
    """
    Extract poses and calibration information from the KITTI dataset.

    This function processes the OXTS data (GPS/IMU) and extracts the
    pose information (translation and rotation) for each frame. It also
    retrieves the calibration information (transformation matrices and focal length)
    required for further processing.

    Args:
        basedir (str): The base directory containing the KITTI dataset.
        oxts_path_tracking (str, optional): Path to the OXTS data file for tracking sequences.
            If not provided, the function will look for OXTS data in the basedir.
        selected_frames (list, optional): A list of frame indices to process.
            If not provided, all frames in the dataset will be processed.

    Returns:
        tuple: A tuple containing the following elements:
            poses (np.array): An array of 4x4 pose matrices representing the vehicle's
                position and orientation for each frame (IMU pose).
            calibrations (dict): A dictionary containing the transformation matrices
                and focal length obtained from the calibration files.
            focal (float): The focal length of the left camera.
    """

    def oxts_to_pose(oxts):
        """
        OXTS (Oxford Technical Solutions) data typically refers to the data generated by an Inertial and GPS Navigation System (INS/GPS) that is used to provide accurate position, orientation, and velocity information for a moving platform, such as a vehicle. In the context of the KITTI dataset, OXTS data is used to provide the ground truth for the vehicle's trajectory and 6 degrees of freedom (6-DoF) motion, which is essential for evaluating and benchmarking various computer vision and robotics algorithms, such as visual odometry, SLAM, and object detection.

        The OXTS data contains several important measurements:

        1. Latitude, longitude, and altitude: These are the global coordinates of the moving platform.
        2. Roll, pitch, and yaw (heading): These are the orientation angles of the platform, usually given in Euler angles.
        3. Velocity (north, east, and down): These are the linear velocities of the platform in the local navigation frame.
        4. Accelerations (ax, ay, az): These are the linear accelerations in the platform's body frame.
        5. Angular rates (wx, wy, wz): These are the angular rates (also known as angular velocities) of the platform in its body frame.

        In the KITTI dataset, the OXTS data is stored as plain text files with each line corresponding to a timestamp. Each line in the file contains the aforementioned measurements, which are used to compute the ground truth trajectory and 6-DoF motion of the vehicle. This information can be further used for calibration, data synchronization, and performance evaluation of various algorithms.
        """
        poses = []

        def latlon_to_mercator(lat, lon, s):
            """
            Converts latitude and longitude coordinates to Mercator coordinates (x, y) using the given scale factor.

            The Mercator projection is a widely used cylindrical map projection that represents the Earth's surface
            as a flat, rectangular grid, distorting the size of geographical features in higher latitudes.
            This function uses the scale factor 's' to control the amount of distortion in the projection.

            Args:
                lat (float): Latitude in degrees, range: -90 to 90.
                lon (float): Longitude in degrees, range: -180 to 180.
                s (float): Scale factor, typically the cosine of the reference latitude.

            Returns:
                list: A list containing the Mercator coordinates [x, y] in meters.
            """
            r = 6378137.0  # the Earth's equatorial radius in meters
            x = s * r * ((np.pi * lon) / 180)
            y = s * r * np.log(np.tan((np.pi * (90 + lat)) / 360))
            return [x, y]

        # Compute the initial scale and pose based on the selected frames
        if selected_frames is None:
            lat0 = oxts[0][0]
            scale = np.cos(lat0 * np.pi / 180)
            pose_0_inv = None
        else:
            oxts0 = oxts[selected_frames[0][0]]
            lat0 = oxts0[0]
            scale = np.cos(lat0 * np.pi / 180)

            pose_i = np.eye(4)

            [x, y] = latlon_to_mercator(oxts0[0], oxts0[1], scale)
            z = oxts0[2]
            translation = np.array([x, y, z])
            rotation = get_rotation(oxts0[3], oxts0[4], oxts0[5])
            pose_i[:3, :] = np.concatenate([rotation, translation[:, None]], axis=1)
            pose_0_inv = invert_transformation(pose_i[:3, :3], pose_i[:3, 3])

        # Iterate through the OXTS data and compute the corresponding pose matrices
        for oxts_val in oxts:
            pose_i = np.zeros([4, 4])
            pose_i[3, 3] = 1

            [x, y] = latlon_to_mercator(oxts_val[0], oxts_val[1], scale)
            z = oxts_val[2]
            translation = np.array([x, y, z])

            roll = oxts_val[3]
            pitch = oxts_val[4]
            heading = oxts_val[5]
            rotation = get_rotation(roll, pitch, heading)  # (3,3)

            pose_i[:3, :] = np.concatenate([rotation, translation[:, None]], axis=1)  # (4, 4)
            if pose_0_inv is None:
                pose_0_inv = invert_transformation(pose_i[:3, :3], pose_i[:3, 3])

            pose_i = np.matmul(pose_0_inv, pose_i)
            poses.append(pose_i)

        return np.array(poses)

    # If there is no tracking path specified, use the default path
    if oxts_path_tracking is None:
        oxts_path = os.path.join(basedir, "oxts/data")
        oxts = np.array([np.loadtxt(os.path.join(oxts_path, file)) for file in sorted(os.listdir(oxts_path))])
        calibration_path = os.path.dirname(basedir)

        calibrations = calib_from_txt(calibration_path)

        focal = calibrations[4]

        poses = oxts_to_pose(oxts)

    # If a tracking path is specified, use it to load OXTS data and compute the poses
    else:
        oxts_tracking = np.loadtxt(oxts_path_tracking)
        poses = oxts_to_pose(oxts_tracking)  # (n_frames, 4, 4)
        calibrations = None
        focal = None
        # Set velodyne close to z = 0
        # poses[:, 2, 3] -= 0.8

    # Return the poses, calibrations, and focal length
    return poses, calibrations, focal


def invert_transformation(rot, t):
    t = np.matmul(-rot.T, t)
    inv_translation = np.concatenate([rot.T, t[:, None]], axis=1)
    return np.concatenate([inv_translation, np.array([[0.0, 0.0, 0.0, 1.0]])])

# deal with kitti_mot
def get_camera_poses_tracking(poses_velo_w_tracking, tracking_calibration, selected_frames, scene_no=None):
    exp = False
    camera_poses = []

    opengl2kitti = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    start_frame = selected_frames[0]
    end_frame = selected_frames[1]

    #####################
    # Debug Camera offset
    if scene_no == 2:
        yaw = np.deg2rad(0.7)  ## Affects camera rig roll: High --> counterclockwise
        pitch = np.deg2rad(-0.5)  ## Affects camera rig yaw: High --> Turn Right
        # pitch = np.deg2rad(-0.97)
        roll = np.deg2rad(0.9)  ## Affects camera rig pitch: High -->  up
        # roll = np.deg2rad(1.2)
    elif scene_no == 1:
        if exp:
            yaw = np.deg2rad(0.3)  ## Affects camera rig roll: High --> counterclockwise
            pitch = np.deg2rad(-0.6)  ## Affects camera rig yaw: High --> Turn Right
            # pitch = np.deg2rad(-0.97)
            roll = np.deg2rad(0.75)  ## Affects camera rig pitch: High -->  up
            # roll = np.deg2rad(1.2)
        else:
            yaw = np.deg2rad(0.5)  ## Affects camera rig roll: High --> counterclockwise
            pitch = np.deg2rad(-0.5)  ## Affects camera rig yaw: High --> Turn Right
            roll = np.deg2rad(0.75)  ## Affects camera rig pitch: High -->  up
    else:
        yaw = np.deg2rad(0.05)
        pitch = np.deg2rad(-0.75)
        # pitch = np.deg2rad(-0.97)
        roll = np.deg2rad(1.05)
        # roll = np.deg2rad(1.2)

    cam_debug = np.eye(4)
    cam_debug[:3, :3] = get_rotation(roll, pitch, yaw)

    Tr_cam2camrect = tracking_calibration["Tr_cam2camrect"]
    Tr_cam2camrect = np.matmul(Tr_cam2camrect, cam_debug)
    Tr_camrect2cam = invert_transformation(Tr_cam2camrect[:3, :3], Tr_cam2camrect[:3, 3])
    Tr_velo2cam = tracking_calibration["Tr_velo2cam"]
    Tr_cam2velo = invert_transformation(Tr_velo2cam[:3, :3], Tr_velo2cam[:3, 3])

    camera_poses_imu = []
    for cam in camera_ls:
        Tr_camrect2cam_i = tracking_calibration["Tr_camrect2cam0" + str(cam)]
        Tr_cam_i2camrect = invert_transformation(Tr_camrect2cam_i[:3, :3], Tr_camrect2cam_i[:3, 3])
        # transform camera axis from kitti to opengl for nerf:
        cam_i_camrect = np.matmul(Tr_cam_i2camrect, opengl2kitti)
        cam_i_cam0 = np.matmul(Tr_camrect2cam, cam_i_camrect)
        cam_i_velo = np.matmul(Tr_cam2velo, cam_i_cam0)

        cam_i_w = np.matmul(poses_velo_w_tracking, cam_i_velo)
        camera_poses_imu.append(cam_i_w)

    for i, cam in enumerate(camera_ls):
        for frame_no in range(start_frame, end_frame + 1):
            camera_poses.append(camera_poses_imu[i][frame_no])

    return np.array(camera_poses)

# deal with kitti_mot
def get_scene_images_tracking(tracking_path, sequence, selected_frames):
    [start_frame, end_frame] = selected_frames
    img_name = []
    sky_name = []

    left_img_path = os.path.join(os.path.join(tracking_path, "image_02"), sequence)
    right_img_path = os.path.join(os.path.join(tracking_path, "image_03"), sequence)

    left_sky_path = os.path.join(os.path.join(tracking_path, "sky_02"), sequence)
    right_sky_path = os.path.join(os.path.join(tracking_path, "sky_03"), sequence)

    for frame_dir in [left_img_path, right_img_path]:
        for frame_no in range(len(os.listdir(left_img_path))):
            if start_frame <= frame_no <= end_frame:
                frame = sorted(os.listdir(frame_dir))[frame_no]
                fname = os.path.join(frame_dir, frame)
                img_name.append(fname)

    for frame_dir in [left_sky_path, right_sky_path]:
        for frame_no in range(len(os.listdir(left_sky_path))):
            if start_frame <= frame_no <= end_frame:
                frame = sorted(os.listdir(frame_dir))[frame_no]
                fname = os.path.join(frame_dir, frame)
                sky_name.append(fname)

    return img_name, sky_name

def rotation_matrix(a, b):
    """Compute the rotation matrix that rotates vector a to vector b.

    Args:
        a: The vector to rotate.
        b: The vector to rotate to.
    Returns:
        The rotation matrix.
    """
    a = a / torch.linalg.norm(a)
    b = b / torch.linalg.norm(b)
    v = torch.cross(a, b)
    c = torch.dot(a, b)
    # If vectors are exactly opposite, we add a little noise to one of them
    if c < -1 + 1e-8:
        eps = (torch.rand(3) - 0.5) * 0.01
        return rotation_matrix(a + eps, b)
    s = torch.linalg.norm(v)
    skew_sym_mat = torch.Tensor(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    return torch.eye(3) + skew_sym_mat + skew_sym_mat @ skew_sym_mat * ((1 - c) / (s**2 + 1e-8))

def auto_orient_and_center_poses(
    poses,
):
    """
    From nerfstudio
    https://github.com/nerfstudio-project/nerfstudio/blob/8e0c68754b2c440e2d83864fac586cddcac52dc4/nerfstudio/cameras/camera_utils.py#L515
    """
    origins = poses[..., :3, 3]
    mean_origin = torch.mean(origins, dim=0)
    translation = mean_origin
    up = torch.mean(poses[:, :3, 1], dim=0)
    up = up / torch.linalg.norm(up)
    rotation = rotation_matrix(up, torch.Tensor([0, 0, 1]))
    transform = torch.cat([rotation, rotation @ -translation[..., None]], dim=-1)
    oriented_poses = transform @ poses
    return oriented_poses, transform

def _convert_to_float(val):
    try:
        v = float(val)
        return v
    except:
        if val == 'True':
            return 1
        elif val == 'False':
            return 0
        else:
            ValueError('Is neither float nor boolean: ' + val)


def _get_kitti_information(path):
     f = open(path, 'r')
     c = f.read()
     c = c.split("\n", 1)[1]
     return np.array([[_convert_to_float(j) for j in i.split(' ')] for i in c.splitlines()])

# not used
def _get_scene_objects(basedir):
    """

    Args:
        basedir:

    Returns:
        objct pose:
            rame cameraID trackID
            alpha width height length
            world_space_X world_space_Y world_space_Z
            rotation_world_space_y rotation_world_space_x rotation_world_space_z
            camera_space_X camera_space_Y camera_space_Z
            rotation_camera_space_y rotation_camera_space_x rotation_camera_space_z
            is_moving
        vehicles_meta:
            trackID
            onehot encoded Label
            onehot encoded vehicle model
            onehot encoded color
            3D bbox dimension (length, height, width)
        max_obj:
            Maximum number of objects in a single frame
        bboxes_by_frame:
            2D bboxes
    """
    object_pose = _get_kitti_information(os.path.join(basedir, 'pose.txt'))
    print('Loading poses from: ' + os.path.join(basedir, 'pose.txt'))
    bbox = _get_kitti_information(os.path.join(basedir, 'bbox.txt'))
    print('Loading bbox from: ' + os.path.join(basedir, 'bbox.txt'))
    info = open(os.path.join(basedir, 'info.txt')).read()
    print('Loading info from: ' + os.path.join(basedir, 'info.txt'))
    info = info.splitlines()[1:]

    # Creates a dictionary which label and model for each track_id
    vehicles_meta = {}

    for i, vehicle in enumerate(info):
        # Vehicle
        # label = np.zeros([len(_sem2label)])
        # model = np.zeros([len(_sem2model)])
        # color = np.zeros([len(_sem2color)])
        vehicle = vehicle.split()  # Ignores colour for now

        # label[_sem2label[vehicle[1]]] = 1
        # model[_sem2model[vehicle[2]]] = 1
        # color[_sem2color[vehicle[3]]] = 1

        label = np.array([_sem2label[vehicle[1]]])

        track_id = np.array([int(vehicle[0])])

        # width height length
        vehicle_dim = object_pose[np.where(object_pose[:, 2] == track_id), :][0, 0, 4:7]
        # For vkitti2 dimensions are defined: width height length
        # To Match vehicle axis xyz swap to length, height, width
        vehicle_dim = vehicle_dim[[2, 1, 0]]

        # vehicle = np.concatenate((np.concatenate((np.concatenate((track_id, label)), model)), color))
        vehicle = np.concatenate([track_id, vehicle_dim])
        vehicle = np.concatenate([vehicle, label])
        vehicles_meta[int(track_id)] = vehicle

    # Get the maximum number of objects in a single frame to define networks
    # input size for the specific scene if objects are used
    max_obj = 0
    f = 0
    c = 0
    count = 0
    for obj in object_pose[:,:2]:
        count += 1
        if not obj[0] == f or obj[1] == c:
            f = obj[0]
            c = obj[1]
            if count > max_obj:
                max_obj = count
            count = 0

    # Add to object_pose if the object is moving between the current and the next frame
    # TODO: Use if moving information to decide if an Object is static or dynamic across the whole scene!!
    object_pose = np.column_stack((object_pose, bbox[:, -1]))

    # Store 2D bounding boxes of frames
    bboxes_by_frame = []
    last_frame = bbox[-1, 0].astype(np.int32)
    for cam in range(2):
        for i in range(last_frame + 1):
            bbox_at_i = np.squeeze(bbox[np.argwhere(bbox[:, 0] == i), :7])
            bboxes_by_frame.append(bbox_at_i[np.argwhere(bbox_at_i[:, 1] == cam), 3:7])


    return object_pose, vehicles_meta, max_obj, bboxes_by_frame

# Get points from depth image under world coordinates
def inverse_projection(depth, intrinsics, c2w, valid_depth_mask=None):
    """
    Performs inverse projection from depth map to 3D points in world coordinates.
    Args:
        depth (ndarray): Depth map. Shape (H, W).
        intrinsics (tuple): Camera intrinsics (fx, fy, cx, cy).
        c2w (ndarray): Camera-to-world transformation matrix. Shape (4, 4).
        valid_depth_mask (ndarray, optional): Mask for valid depth values. Shape (H, W).
    Returns:
        ndarray: 3D points in world coordinates.
    """
    H, W = depth.shape
    fx, fy, cx, cy = intrinsics

    # Create grid of pixel coordinates
    x = np.arange(W) - cx
    y = np.arange(H) - cy
    xx, yy = np.meshgrid(x, y)

    # If there is a valid depth mask, apply it
    if valid_depth_mask is not None:
        xx = xx[valid_depth_mask]
        yy = yy[valid_depth_mask]
        z = depth[valid_depth_mask]
    else:
        z = depth

    # Project pixels to 3D camera coordinates
    xx = xx * z / fx
    yy = yy * z / fy

    # Stack coordinates and add a homogeneous coordinate
    points = np.stack([xx, yy, z], axis=-1)
    points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)

    # Transform to world coordinates
    points = np.matmul(c2w, points.reshape(-1, 4).T).T

    # Remove the homogeneous coordinate before returning
    return points[:, :3]


def voxel_downsample(pointcloud, pointcloud_timestamp,voxel_size):
    print(f'before:{pointcloud.shape}')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud[:, :3])

    # 进行体素下采样
    pcd_downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    downsampled_points = np.asarray(pcd_downsampled.points)

    if downsampled_points.size == 0:
        raise ValueError("Downsampled point cloud is empty. Adjust voxel size or check input data.")

    kdtree = o3d.geometry.KDTreeFlann(pcd)

    indices = []
    for point in downsampled_points:
        _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
        indices.append(idx[0])

    indices = np.array(indices)

    downsampled_timestamps = pointcloud_timestamp[indices]

    pointcloud = np.hstack((downsampled_points, pointcloud[indices, 3:]))
    pointcloud_timestamp = downsampled_timestamps

    print(f'end:{pointcloud.shape}')

    return pointcloud,pointcloud_timestamp


def readVKittiInfo(args):
    cam_infos = []
    points = []
    points_time = []

    basedir = args.source_path
    extrinsic = _get_kitti_information(os.path.join(basedir, 'extrinsic.txt')) # (n_frames * n_cameras, 18)
    intrinsic = _get_kitti_information(os.path.join(basedir, 'intrinsic.txt')) # (n_frames * n_cameras, 6)
    # object_pose, object_meta, max_objects_per_frame, bboxes = _get_scene_objects(basedir)

    poses = []
    extrinsics = []
    frame_id = []
    
    
    image_filenames = []
    depth_filenames = []
    seg_filenames = []
    sky_filenames = []
    
    ply_paths=[]
    bin_paths=[]

    rgb_dir = os.path.join(basedir, 'frames/rgb')
    depth_dir = os.path.join(basedir, 'frames/depth')
    segm_dir = os.path.join(basedir, 'frames/classSegmentation')
    sky_mask_dir = os.path.join(basedir, 'frames/sky_mask')
    n_cam = len(camera_ls) # [0,1]
    
    selected_frames = [args.start_frame, args.end_frame]
 
    time_duration = [0, 1]
    
    for camera in camera_ls:
        camera_dir = 'Camera_' + str(camera)
        frame_folder = os.path.join(rgb_dir, camera_dir)
        seg_folder = os.path.join(segm_dir, camera_dir)
        depth_folder = os.path.join(depth_dir, camera_dir)
        sky_mask_folder = os.path.join(sky_mask_dir, camera_dir)

        os.makedirs(sky_mask_folder, exist_ok=True)
        # TODO: Check mismatching numbers of poses and Images like in loading script for llf
        for frame in sorted(os.listdir(frame_folder)):
            if frame.endswith('.jpg'):
                frame_num = int(frame.split('rgb_')[1].split('.jpg')[0])

                if selected_frames[0] <= frame_num <= selected_frames[1]:
                    fname = os.path.join(frame_folder, frame)
                    image_filenames.append(fname)
                    # rgb_img = imageio.imread(fname)
                    # imgs.append(rgb_img)

                    depth_path = os.path.join(depth_folder, 'depth_' + '{:05}'.format(frame_num) + '.png')
                    depth_filenames.append(depth_path)


                    # depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 100.
                    # depths.append(depth)

                    seg_frame = 'classgt_' + '{:05}'.format(frame_num) + '.png'
                    seg_gt_name = os.path.join(seg_folder, seg_frame)
                    
                    seg_filenames.append(seg_gt_name)

                    # if os.path.exists(seg_gt_name):
                        
                    #     class_segm_img = (np.asarray(Image.open(seg_gt_name)))
                    #     segm.append(class_segm_img)

                    sky_mask_path = os.path.join(sky_mask_folder, 'sky_mask_' + '{:05}'.format(frame_num) + '.png')

                    # Create sky mask from depth image if sky mask simage does not exist
                    if not os.path.exists(sky_mask_path):
                        depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                        valid_mask = (depth_img == 65535)  
                        valid_mask_uint8 = (valid_mask.astype(np.uint8)) * 255
                        cv2.imwrite(sky_mask_path, valid_mask_uint8)
                    sky_filenames.append(sky_mask_path)
                        
                    ext = extrinsic[frame_num * n_cam: frame_num * n_cam + n_cam, :][camera][2:]
                    ext = np.reshape(ext, (-1, 4))
                    extrinsics.append(ext)

                    # Get camera pose and location from extrinsics
                    pose = np.zeros([4, 4])
                    pose[3, 3] = 1
                    R = np.transpose(ext[:3, :3])
                    t = -ext[:3, -1]

                    # Camera position described in world coordinates
                    pose[:3, -1] = np.matmul(R, t)
                    # Match OpenGL definition of Z
                    pose[:3, :3] = np.matmul(np.eye(3), np.matmul(np.eye(3), R))
                    # Rotate pi around Z
                    poses.append(pose)

                    # pose: camera to world
                    frame_id.append([frame_num, camera, 0])

    poses = np.array(poses).astype(np.float32) # (n_frames, 4, 4)
    
    intrinsics = intrinsic[0, 2:] # (n_cameras, 4)
    focal_X, focal_Y = intrinsics[0], intrinsics[1]
    cx, cy = intrinsics[2], intrinsics[3]

    c2ws = poses

    for idx in tqdm(range(len(c2ws)), desc="Loading data"):
        c2w = c2ws[idx]
        w2c = np.linalg.inv(c2w)
        image_path = image_filenames[idx]
        image_name = os.path.basename(image_path)[:-4]

        sky_path = sky_filenames[idx]
        im_data = Image.open(image_path)
        W, H = im_data.size
        image = np.array(im_data) / 255.

        sky_mask = cv2.imread(sky_path)

        timestamp = time_duration[0] + (time_duration[1] - time_duration[0]) * (idx % (len(c2ws) // 2)) / (len(c2ws) // 2 - 1)
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        depth_map = cv2.imread(depth_filenames[idx], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        valid_mask =~(depth_map==65535)
        depth_map = depth_map / 100.0
        point_xyz_world = inverse_projection(depth_map, intrinsics, c2w, valid_mask)
        # print(point_xyz_world) 
        # THE QUESTION IS TO MOVE POINTCLOUDS TO (0,0,0) AND ROTATE THE COORDINATES TO ALIGN THE LIDAR CLOUDS
        depth_map = np.where(valid_mask, depth_map, 0.0)

        points.append(point_xyz_world)
        

        point_time = np.full_like(point_xyz_world[:, :1], timestamp)
        points_time.append(point_time)


        if args.voxel_downsample:
            points_now=np.concatenate(points,axis=0)
            points_time_now=np.concatenate(points_time,axis=0)
            if points_now.shape[0]>args.num_pts:
                points_now,points_time_now=voxel_downsample(points_now,points_time_now,args.voxel_size)
                print(f"after:")
                print(points_now.shape,points_time_now.shape)
                points=[]
                points_time=[]
                points.append(points_now)
                points_time.append(points_time_now)

        

        
        
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T,
                                    image=image,
                                    image_path=image_filenames[idx], image_name=image_filenames[idx],
                                    width=W, height=H, timestamp=timestamp,
                                    fx=focal_X, fy=focal_Y, cx=cx, cy=cy, sky_mask=sky_mask,
                                    depth_map=depth_map))

        if args.debug_cuda and idx > 5:
            break
    
    pointcloud = np.concatenate(points, axis=0)
    pointcloud_timestamp = np.concatenate(points_time, axis=0)

    # normalize poses
    w2cs = np.zeros((len(cam_infos), 4, 4))
    Rs = np.stack([c.R for c in cam_infos], axis=0)
    Ts = np.stack([c.T for c in cam_infos], axis=0)
    w2cs[:, :3, :3] = Rs.transpose((0, 2, 1))
    w2cs[:, :3, 3] = Ts
    w2cs[:, 3, 3] = 1
    c2ws = unpad_poses(np.linalg.inv(w2cs))
    c2ws, transform, scale_factor = transform_poses_pca(c2ws, fix_radius=args.fix_radius)
    c2ws = pad_poses(c2ws)
    for idx, cam_info in enumerate(tqdm(cam_infos, desc="Transform data")):
        c2w = c2ws[idx]
        w2c = np.linalg.inv(c2w)
        cam_info.R[:] = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        cam_info.T[:] = w2c[:3, 3]
        cam_info.depth_map[:] *= scale_factor
    pointcloud = (np.pad(pointcloud, ((0, 0), (0, 1)), constant_values=1) @ transform.T)[:, :3]
    pointcloud = (np.pad(pointcloud, ((0, 0), (0, 1)), constant_values=1))[:, :3]

    point_norms=np.linalg.norm(pointcloud[:,:3],axis=1)
    norm_mask=point_norms<80*scale_factor
    
    pointcloud=pointcloud[norm_mask]
    pointcloud_timestamp=pointcloud_timestamp[norm_mask]
    print(pointcloud.shape)

    # 全局体素下采样，弃用

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pointcloud[:, :3])

    # voxel_size = 0.005  # 体素大小
    # pcd_downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    # downsampled_points = np.asarray(pcd_downsampled.points)

    # if downsampled_points.size == 0:
    #     raise ValueError("Downsampled point cloud is empty. Adjust voxel size or check input data.")

    # kdtree = o3d.geometry.KDTreeFlann(pcd)

    # indices = []
    # for point in downsampled_points:
    #     _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
    #     indices.append(idx[0])

    # indices = np.array(indices)

    # downsampled_timestamps = pointcloud_timestamp[indices]

    # pointcloud = np.hstack((downsampled_points, pointcloud[indices, 3:]))
    # pointcloud_timestamp = downsampled_timestamps

    # print(pointcloud.shape)


    if pointcloud.shape[0] > args.num_pts:
        indices = np.random.choice(pointcloud.shape[0], args.num_pts, replace=True)
        pointcloud = pointcloud[indices]
        pointcloud_timestamp = pointcloud_timestamp[indices]

    print(pointcloud.shape)
    if args.eval:
        num_frame = len(cam_infos)//2
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx % num_frame + 1) % args.testhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx % num_frame + 1) % args.testhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    # for kitti have some static ego videos, we dont calculate radius here
    nerf_normalization = getNerfppNorm(train_cam_infos)
    nerf_normalization['radius'] = 1
    if not args.voxel_downsample:
        ply_path = os.path.join(args.source_path, "points3d.ply")
    else:
        ply_path = os.path.join(args.source_path, "points3d_voxel.ply")
    # TODO: for debug
    rgbs = np.random.random((pointcloud.shape[0], 3))
    storePly(ply_path, pointcloud, rgbs, pointcloud_timestamp)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    if args.visualize_pointcloud:
        ply_paths.append(ply_path)
        visualize_multiple_point_clouds(ply_paths,bin_paths)

    time_interval = (time_duration[1] - time_duration[0]) / (frame_num - 1)


    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           time_interval=time_interval)

    return scene_info

def visualize_point_cloud(points):
    """
    Visualize a point cloud using Open3D.
    
    Args:
        points (np.ndarray): The point cloud as an (N, 3) array.
    """
    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Optional: Set a uniform color for the point cloud (e.g., white)
    pcd.paint_uniform_color([1, 0, 1])
    
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

def load_point_cloud_from_bin(bin_file_path):
    """
    Load a point cloud from a .bin file.
    
    Args:
        bin_file_path (str): Path to the .bin file.
        
    Returns:
        np.ndarray: The point cloud as an (N, 3) array.
    """
    # Load the binary file
    point_cloud = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)
    # Extract XYZ coordinates
    points = point_cloud[:, :3]
    return points

def load_point_cloud_from_ply(ply_file_path):
    """
    Load a point cloud from a .ply file using Open3D.
    
    Args:
        ply_file_path (str): Path to the .ply file.
        
    Returns:
        o3d.geometry.PointCloud: The loaded point cloud.
    """
    pcd = o3d.io.read_point_cloud(ply_file_path)
    return pcd

def visualize_multiple_point_clouds(ply_files, bin_files, axis_size=1.0, point_size=1.0):
    """
    Visualize multiple PLY and BIN point clouds using Open3D with a large coordinate frame and adjustable point size.
    
    Args:
        ply_files (list of str): List of paths to .ply files.
        bin_files (list of str): List of paths to .bin files.
        axis_size (float): Size of the coordinate axis.
        point_size (float): Size of the points in the point cloud.
    """
    pcd_list = []
    for i, ply_file in enumerate(ply_files):
        pcd = load_point_cloud_from_ply(ply_file)
        
        points = np.asarray(pcd.points)
        
        # 将 y 和 z 坐标变为负数
        points[:, 1] = -points[:, 1]  
        points[:, 2] = -points[:, 2]  
        pcd.points = o3d.utility.Vector3dVector(points)

        max_norm = np.max(np.linalg.norm(points, axis=1))
        print(f"PLY File: {ply_file}, Max Norm: {max_norm}")

        color = np.random.rand(3)
        pcd.paint_uniform_color(color)
        
        pcd_list.append(pcd)
    

    for i, bin_file in enumerate(bin_files):
        points = load_point_cloud_from_bin(bin_file)

        max_norm = np.max(np.linalg.norm(points, axis=1))
        print(f"BIN File: {bin_file}, Max Norm: {max_norm}")
        

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        color = np.random.rand(3)
        pcd.paint_uniform_color(color)
        
        pcd_list.append(pcd)
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0, 0, 0])
    pcd_list.append(coordinate_frame)
    

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for pcd in pcd_list:
        vis.add_geometry(pcd)

    render_option = vis.get_render_option()
    render_option.point_size = point_size
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, default="/home/chengwei/Desktop/summer_research/SUDS/depth_recover/vkitti2/Scene01/clone")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=10)
    parser.add_argument("--frame_interval", type=int, default=10)
    parser.add_argument("--time_duration", type=list, default=[0, 1])
    parser.add_argument("--num_pts", type=int, default=100000)
    parser.add_argument("--fix_radius", type=float, default=0)
    parser.add_argument("--debug_cuda", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--testhold", type=int, default=10)
    parser.add_argument("--voxel_downsample", action="store_true", default=False, help="Enable voxel downsampling.")
    parser.add_argument("--voxel_size",type=float,default=0.15)
    args = parser.parse_args()

    scene_info = readVKittiInfo(args)

    ply_files = ['/home/chengwei/Desktop/summer_research/SUDS/depth_recover/vkitti2/Scene01/clone/points3d.ply']
    # Get a list of all .bin files in the directory
    bin_files = ['/home/chengwei/Desktop/summer_research/SUDS/depth_recover/kitti_mot/training/velodyne/0001/000000.bin']

    visualize_multiple_point_clouds(ply_files, bin_files)