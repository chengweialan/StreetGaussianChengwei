"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time

import cv2
import numpy as np
import os
import fusion


if __name__ == "__main__":
  # ======================================================================================================== #
  # (Optional) This is an example of how to compute the 3D bounds
  # in world coordinates of the convex hull of all camera view
  # frustums in the dataset
  # ======================================================================================================== #
  print("Estimating voxel volume bounds...")
  n_frames = 50
  cam_nums=3
  tsdf_path='eval_output/waymo_reconstruction/022_dynamic_normal_v_smooth/eval/tsdf'
  vol_bnds = np.zeros((3,2))
  vol_bnds[:,0]=-5
  vol_bnds[:,1]=5
  for i in range(n_frames):
    # Read depth image and camera pose
    for cam_id in range(cam_nums):
        cam_intr=np.load(os.path.join(tsdf_path,f'{i:02d}{cam_id}_intrinsics.npy'))
        depth_im=np.load(os.path.join(tsdf_path,f'{i:02d}{cam_id}_depth.npy')).squeeze()
        # depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im > 10] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
        cam_pose = np.load(os.path.join(tsdf_path,f'{i:02d}{cam_id}_c2w.npy'))  # 4x4 rigid transformation matrix

        # Compute camera view frustum and extend convex hull
        view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
        # vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1),-10)
        # vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1),10)
  # ======================================================================================================== #

  # ======================================================================================================== #
  # Integrate
  # ======================================================================================================== #
  # Initialize voxel volume
  print("Initializing voxel volume...")
  tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.02)

  # Loop through RGB-D images and fuse them together
  t0_elapse = time.time()
  for i in range(n_frames):
    for cam_id in range(cam_nums):
        print("Fusing frame %d/%d"%(i+1, n_frames))

        # Read RGB-D image and camera pose
        color_image = cv2.cvtColor(cv2.imread(os.path.join(tsdf_path,f'{i:02d}{cam_id}.png')), cv2.COLOR_BGR2RGB)
        depth_im=np.load(os.path.join(tsdf_path,f'{i:02d}{cam_id}_depth.npy')).squeeze()
        # depth_im /= 1000.
        # depth_im[depth_im == 65.535] = 0
        cam_pose = np.load(os.path.join(tsdf_path,f'{i:02d}{cam_id}_c2w.npy'))

        # Integrate observation into voxel volume (assume color aligned with depth)
        tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

  fps = n_frames / (time.time() - t0_elapse)
  print("Average FPS: {:.2f}".format(fps))

  # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving mesh to mesh.ply...")
  verts, faces, norms, colors = tsdf_vol.get_mesh()
  fusion.meshwrite("mesh.ply", verts, faces, norms, colors)

  # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving point cloud to pc.ply...")
  point_cloud = tsdf_vol.get_point_cloud()
  fusion.pcwrite("pc.ply", point_cloud)