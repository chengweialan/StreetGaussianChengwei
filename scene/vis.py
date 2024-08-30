import os
import cv2
import sys
import open3d as o3d
# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import imageio.v2 as imageio
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from scene.scene_utils import CameraInfo, SceneInfo, getNerfppNorm, fetchPly, storePly
from pathlib import Path

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
    print(point_cloud.shape)
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
    scale=1
    # Load and prepare PLY point clouds
    for i, ply_file in enumerate(ply_files):
        pcd = load_point_cloud_from_ply(ply_file)
        
        # 获取点云的坐标数组
        points = np.asarray(pcd.points)
        
        # 将 y 和 z 坐标变为负数
        points[:, 1] = -points[:, 1]  # 反转 y 坐标
        points[:, 2] = -points[:, 2]  # 反转 z 坐标
        points[:,:3]*=scale
        # 将修改后的坐标重新赋值回点云
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 计算并打印点云模长的最大值
        max_norm = np.max(np.linalg.norm(points, axis=1))
        print(f"PLY File: {ply_file}, Max Norm: {max_norm}")
        
        # 为每个 PLY 点云设置一个唯一的颜色（可选）
        color = np.random.rand(3)
        pcd.paint_uniform_color(color)
        
        # 将点云对象添加到列表中
        pcd_list.append(pcd)
    
    # Load and prepare BIN point clouds
    for i, bin_file in enumerate(bin_files):
        points = load_point_cloud_from_bin(bin_file)
        
        # Compute and print the maximum norm of the point cloud
        max_norm = np.max(np.linalg.norm(points, axis=1))
        print(f"BIN File: {bin_file}, Max Norm: {max_norm}")
        
        # Create an Open3D PointCloud object for BIN files
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Set a unique color for each BIN point cloud (optional)
        color = np.random.rand(3)
        pcd.paint_uniform_color(color)
        
        pcd_list.append(pcd)
    
    # Create a coordinate frame (axis) with a custom size
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.5, origin=[0, 0, 0])
    pcd_list.append(coordinate_frame)
    
    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add all geometries (point clouds and coordinate frame) to the visualizer
    for pcd in pcd_list:
        vis.add_geometry(pcd)
    
    # Get the render option and set the point size
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()



def parse_kitti_calibration_file(file_path):
    calibration_data = {}
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            key_value = line.split(':')
            if len(key_value) == 2:
                key = key_value[0].strip()
                values = np.array([float(x) for x in key_value[1].strip().split()])
                
                if key.startswith('P'):
                    calibration_data[key] = values.reshape(3, 4)
                elif key.startswith('R_rect'):
                    # R_rect 是 3x3 的矩阵
                    calibration_data[key] = values.reshape(3, 3)
                elif key.startswith('Tr'):
                    # Tr 是 3x4 的矩阵，但需要转换为 4x4 的齐次矩阵
                    calibration_data[key] = np.vstack([values.reshape(3, 4), [0, 0, 0, 1]])
    
    return calibration_data

calib_path='/home/chengwei/Desktop/summer_research/SUDS/depth_recover/kitti_mot/training/calib/0001.txt'
calib_data=parse_kitti_calibration_file(calib_path)
lidar2cam=calib_data['P0']

ply_files=['/home/chengwei/Desktop/summer_research/SUDS/depth_recover/vkitti2/Scene01/clone/points3d.ply']
ply_path='/home/chengwei/Desktop/summer_research/SUDS/depth_recover/vkitti2/Scene01/clone/ply'
# for idx in range(0,1):
#     path=os.path.join(ply_path,f'{idx}_depth.ply')
#     ply_files.append(path)
# ply_files.append('/home/chengwei/Desktop/summer_research/SUDS/depth_recover/vkitti2/Scene01/clone/ply/camera_before.ply')
# ply_files.append('/home/chengwei/Desktop/summer_research/SUDS/depth_recover/vkitti2/Scene01/clone/ply/camera_after.ply')

# Get a list of all .bin files in the directory
bin_files = []
bin_path='/home/chengwei/Desktop/summer_research/SUDS/depth_recover/kitti_mot/training/velodyne/0001'
for idx in range(0,10):
    path=os.path.join(bin_path,f"{idx:06d}.bin")
    bin_files.append(path)
bin_files = []
visualize_multiple_point_clouds(ply_files, bin_files,point_size=1.0)
