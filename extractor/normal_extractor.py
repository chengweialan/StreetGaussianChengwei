import open3d as o3d
import os
from typing import Optional,List,Tuple,Literal
import vdbfusion
import torch
import numpy as np
import random
from torch import Tensor
from extractors.dn_utils import (
    get_colored_points_from_depth,
    get_means3d_backproj,
    project_pix,
)
import cv2
from tqdm import tqdm

# def TSDFFusion(tsdf_path:str,output_dir:str,voxel_size: float = 0.05,sdf_truc: float = 0.2,total_points: int = 8_000_000,target_triangles: Optional[int] = None,num_frames=50,cam_nums=3):
#     """
#     Backproject depths and run TSDF fusion
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     TSDFvolume = vdbfusion.VDBVolume(
#         voxel_size=voxel_size, sdf_trunc=sdf_truc, space_carving=True
#     )

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     with torch.no_grad():
#         samples_per_frame = (total_points + num_frames) // (num_frames)
#         print("samples per frame: ", samples_per_frame)
#         points = []
#         colors = []
#         for i in tqdm(range(num_frames), desc="Processing frames"):  
#             for cam_id in range(cam_nums):
#                 mask = None
#                 cam_intr=np.load(os.path.join(tsdf_path,f'{i:02d}{cam_id}_intrinsics.npy'))
#                 depth_map=np.load(os.path.join(tsdf_path,f'{i:02d}{cam_id}_depth.npy')).squeeze()
#                 depth_map[depth_map > 10] = 0
                

#                 depth_map = torch.tensor(depth_map, dtype=torch.float32, device=device)

#                 c2w = torch.tensor(np.load(os.path.join(tsdf_path, f'{i:02d}{cam_id}_c2w.npy')), dtype=torch.float, device=device)


#                 c2w = c2w @ torch.diag(
#                     torch.tensor([1, -1, -1, 1], device=c2w.device, dtype=torch.float)
#                 )
#                 c2w = c2w[:3, :4]
#                 H, W = depth_map.shape
#                 color_image = cv2.cvtColor(cv2.imread(os.path.join(tsdf_path,f'{i:02d}{cam_id}.png')), cv2.COLOR_BGR2RGB)

#                 color_image = torch.tensor(color_image, dtype=torch.float32, device=device)

#                 indices = random.sample(range(H * W), samples_per_frame)

#                 xyzs, rgbs = get_colored_points_from_depth(
#                     depths=depth_map,
#                     rgbs=color_image,
#                     fx=cam_intr[0,0],
#                     fy=cam_intr[1,1],
#                     cx=cam_intr[0,2],  # type: ignore
#                     cy=cam_intr[1,2],  # type: ignore
#                     img_size=(W, H),
#                     c2w=c2w,
#                     mask=indices,
#                 )
#                 # xyzs = xyzs[mask.view(-1,1)[...,0]]

#                 min_val = -5
#                 max_val = 5

#                 # 创建一个mask，选择 x, y, z 都在 [-5, 5] 范围内的点
#                 mask = (xyzs[:, 0] >= min_val) & (xyzs[:, 0] <= max_val) & \
#                     (xyzs[:, 1] >= min_val) & (xyzs[:, 1] <= max_val) & \
#                     (xyzs[:, 2] >= min_val) & (xyzs[:, 2] <= max_val)

#                 # 根据 mask 筛选 points、colors、normals
#                 xyzs= xyzs[mask]
#                 rgbs = rgbs[mask]


#                 points.append(xyzs)
#                 colors.append(rgbs)
                

#                 TSDFvolume.integrate(
#                     xyzs.double().cpu().numpy(),
#                     extrinsic=c2w[:3, 3].double().cpu().numpy(),
#                 )

#         vertices, faces = TSDFvolume.extract_triangle_mesh(min_weight=5)

#         mesh = o3d.geometry.TriangleMesh()
#         mesh.vertices = o3d.utility.Vector3dVector(vertices)
#         mesh.triangles = o3d.utility.Vector3iVector(faces)
#         mesh.compute_vertex_normals()
#         colors = torch.cat(colors, dim=0)
#         colors = colors.cpu().numpy()/255.0
#         mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

#         # simplify mesh
#         # if target_triangles is not None:
#         #     mesh = mesh.simplify_quadric_decimation(target_triangles)

        
#         o3d.io.write_triangle_mesh(
#             os.path.join(output_dir,'TSDFfusion_mesh.ply'),
#             mesh,
#         )
#         print(
#             f"Finished computing mesh: {os.path.join(output_dir,'TSDFfusion_mesh.ply')}"
#         )

def find_depth_edges(depth_im, threshold=0.01, dilation_itr=3):
    laplacian_kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=depth_im.dtype, device=depth_im.device
    )
    laplacian_kernel = laplacian_kernel.unsqueeze(0).unsqueeze(0)
    depth_laplacian = (
        F.conv2d(
            (1.0 / (depth_im + 1e-6)).unsqueeze(0).unsqueeze(0).squeeze(-1),
            laplacian_kernel,
            padding=1,
        )
        .squeeze(0)
        .squeeze(0)
        .unsqueeze(-1)
    )

    edges = (depth_laplacian > threshold) * 1.0
    structure_el = laplacian_kernel * 0.0 + 1.0

    dilated_edges = edges
    for i in range(dilation_itr):
        dilated_edges = (
            F.conv2d(
                dilated_edges.unsqueeze(0).unsqueeze(0).squeeze(-1),
                structure_el,
                padding=1,
            )
            .squeeze(0)
            .squeeze(0)
            .unsqueeze(-1)
        )
    dilated_edges = (dilated_edges > 0.0) * 1.0
    return dilated_edges

def pick_indices_at_random(valid_mask, samples_per_frame):
    indices = torch.nonzero(torch.ravel(valid_mask))
    if samples_per_frame < len(indices):
        which = torch.randperm(len(indices))[:samples_per_frame]
        indices = indices[which]
    return torch.ravel(indices)

def DepthAndNormalMapsPoisson(tsdf_path:str,output_dir:str,total_points: int = 2_000_000,normal_method: Literal["density_grad", "normal_maps"] = "normal_maps",use_masks: bool = True,filter_edges_from_depth_maps: bool = False,down_sample_voxel: Optional[float] = None,outlier_removal: bool = False,std_ratio: float = 2.0,edge_threshold: float = 0.004,edge_dilation_iterations: int = 10,poisson_depth: int = 9,num_frames=50,cam_nums=3):
    """
    Idea: backproject depth and normal maps into 3D oriented point cloud -> Poisson
    """

    # total_points: int = 2_000_000
    # """Total target surface samples"""
    # normal_method: Literal["density_grad", "normal_maps"] = "normal_maps"
    # """Normal estimation method"""
    # use_masks: bool = True
    # """If dataset has masks, use these to auto crop gaussians within masked regions."""
    # filter_edges_from_depth_maps: bool = False
    # """Filter out edges when backprojecting from depth maps"""
    # down_sample_voxel: Optional[float] = None
    # """pcd down sample voxel size. Recommended value around 0.005"""
    # outlier_removal: bool = False
    # """Remove outliers"""
    # std_ratio: float = 2.0
    # """Threshold based on STD of the average distances across the point cloud to remove outliers."""
    # edge_threshold: float = 0.004
    # """Threshold for edge detection in depth maps (inverse depth Laplacian, resolution sensitive)"""
    # edge_dilation_iterations: int = 10
    # """Number of morphological dilation iterations for edge detection (swells edges)"""
    # poisson_depth: int = 9
    # """Poisson Octree max depth, higher values increase mesh detail"""


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        samples_per_frame = (total_points + num_frames) // (num_frames)
        print("samples per frame: ", samples_per_frame)
        points = []
        normals = []
        colors = []
        for i in tqdm(range(num_frames), desc="Processing frames"):  
            for cam_id in range(cam_nums):
                mask = None
                cam_intr=np.load(os.path.join(tsdf_path,f'{i:02d}{cam_id}_intrinsics.npy'))
                depth_map=np.load(os.path.join(tsdf_path,f'{i:02d}{cam_id}_depth.npy')).squeeze()
                depth_map[depth_map > 10] = 0
                

                depth_map = torch.tensor(depth_map, dtype=torch.float32, device=device)

                c2w = torch.tensor(np.load(os.path.join(tsdf_path, f'{i:02d}{cam_id}_c2w.npy')), dtype=torch.float, device=device)


                c2w = c2w @ torch.diag(
                    torch.tensor([1, 1, 1, 1], device=c2w.device, dtype=torch.float)
                )
                c2w = c2w[:3, :4]
                H, W = depth_map.shape

                color_image = cv2.cvtColor(cv2.imread(os.path.join(tsdf_path,f'{i:02d}{cam_id}.png')), cv2.COLOR_BGR2RGB)

                color_image = torch.tensor(color_image, dtype=torch.float32, device=device)


                if filter_edges_from_depth_maps:
                    valid_depth = (
                        find_depth_edges(
                            depth_map,
                            threshold=edge_threshold,
                            dilation_itr=edge_dilation_iterations,
                        )
                        < 0.2
                    )
                else:
                    valid_depth = depth_map
                valid_mask = valid_depth

                indices = pick_indices_at_random(valid_mask, samples_per_frame)
                if len(indices) == 0:
                    continue
                xyzs, rgbs = get_colored_points_from_depth(
                    depths=depth_map,
                    rgbs=color_image,
                    fx=cam_intr[0,0],
                    fy=cam_intr[1,1],
                    cx=cam_intr[0,2],
                    cy=cam_intr[1,2],
                    img_size=(W, H),
                    c2w=c2w,
                    mask=indices,
                )
                if normal_method == "normal_maps":
                    # normals to OPENGL
                    normal_map = np.load(os.path.join(tsdf_path,f'{i:02d}{cam_id}_normal.npy')).transpose(1,2,0)
                    h, w, _ = normal_map.shape
                    normal_map = torch.tensor(normal_map, dtype=torch.float32, device=device)
                    normal_map = normal_map.view(-1, 3)
                    normal_map = 2 * normal_map - 1
                    normal_map = normal_map @ torch.diag(
                        torch.tensor(
                            [1, 1, 1], device=normal_map.device, dtype=torch.float
                        )
                    )
                    normal_map = normal_map.view(h, w, 3)
                    # normals to World
                    rot = c2w[:3, :3]
                    normal_map = normal_map.permute(2, 0, 1).reshape(3, -1)
                    normal_map = torch.nn.functional.normalize(normal_map, p=2, dim=0)
                    normal_map = rot @ normal_map
                    normal_map = normal_map.permute(1, 0).reshape(h, w, 3)

                    normal_map = normal_map.view(-1, 3)[indices]
                else:
                    pass #calculate the normal map
                    # grad of density
                    # xyzs, _ = get_means3d_backproj(
                    #     depths=depth_map * 0.99,
                    #     fx=cam_intr[0,0],
                    #     fy=cam_intr[1,1],
                    #     cx=cam_intr[0,2],
                    #     cy=cam_intr[1,2], 
                    #     img_size=(W, H),
                    #     c2w=c2w,
                    #     device=c2w.device,
                    #     # mask=indices,
                    # )
                    # normals = model.get_density_grad(
                    #     samples=xyzs.cuda(), num_closest_gaussians=1
                    # )
                    # viewdirs = -xyzs + c2w[..., :3, 3]
                    # viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
                    # dots = (normals * viewdirs).sum(-1)
                    # negative_dot_indices = dots < 0
                    # normals[negative_dot_indices] = -normals[negative_dot_indices]
                    # normals = normals @ c2w[:3, :3]
                    # normals = normals @ torch.diag(
                    #     torch.tensor(
                    #         [1, -1, -1], device=normals.device, dtype=torch.float
                    #     )
                    # )
                    # normal_map = normals / normals.norm(dim=-1, keepdim=True)
                    # normal_map = (normal_map + 1) / 2

                    # normal_map = outputs["normal"].cpu()
                    # normal_map = normal_map.view(-1, 3)[indices]

                points.append(xyzs)
                colors.append(rgbs)
                normals.append(normal_map)

        points = torch.cat(points, dim=0)
        colors = torch.cat(colors, dim=0)
        normals = torch.cat(normals, dim=0)

        min_val = -5
        max_val = 5

        # 创建一个mask，选择 x, y, z 都在 [-5, 5] 范围内的点
        mask = (points[:, 0] >= min_val) & (points[:, 0] <= max_val) & \
            (points[:, 1] >= min_val) & (points[:, 1] <= max_val) & \
            (points[:, 2] >= min_val) & (points[:, 2] <= max_val)

        # 根据 mask 筛选 points、colors、normals
        points = points[mask]
        colors = colors[mask]
        normals = normals[mask]


        points = points.cpu().numpy()
        normals = normals.cpu().numpy()
        colors = colors.cpu().numpy()/255.0


        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(
            os.path.join(output_dir , f"{num_frames}frames_{total_points}_{poisson_depth}_DepthAndNormalMapsPoisson_pcd.ply"), pcd
        )
        print("Computing Mesh... this may take a while.")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=poisson_depth
        )
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        print("[bold green]:white_check_mark: Computing Mesh")

        print(
            f"Saving Mesh to {os.path.join(output_dir ,f'{num_frames}frames_{total_points}_{poisson_depth}_DepthAndNormalMapsPoisson_poisson_mesh.ply')}"
        )
        o3d.io.write_triangle_mesh(
            os.path.join(output_dir , f"{num_frames}frames_{total_points}_{poisson_depth}_DepthAndNormalMapsPoisson_poisson_mesh.ply"),
            mesh,
        )


tsdf_path='/home/chengwei/Desktop/summer_research/StreetGaussian/eval_output/waymo_reconstruction/022_dynamic_normal_v_smooth/eval/tsdf'
output_dir='/home/chengwei/Desktop/summer_research/StreetGaussian/extractors/output/dn_tsdf_fisiion'
DepthAndNormalMapsPoisson(tsdf_path,output_dir,num_frames=50)