import torch
import os
import open3d as o3d
import numpy as np
from plyfile import PlyData, PlyElement
import argparse
import sys
sys.path.append('/home/chengwei/Desktop/summer_research/StreetGaussian')
from utils.system_utils import mkdir_p
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, get_step_lr_func

def print_ply_properties(ply_file):
    ply_data = PlyData.read(ply_file)
    
    print(ply_data)
    for element in ply_data.elements:
        print(f"Element: {element.name}")
        
        for property in element.properties:
            print(f"  Property: {property.name}")

def construct_list_of_attributes(features_dc, features_rest, scaling, rotation):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(features_dc.shape[1]*features_dc.shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(features_rest.shape[1]*features_rest.shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(scaling.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(rotation.shape[1]):
        l.append('rot_{}'.format(i))
    return l

def save_to_ply(xyz, normal, features_dc, features_rest, opacity, scaling, rotation, output_file):

    mkdir_p(os.path.dirname(output_file))

    num_points = xyz.shape[0]
    xyz=xyz.detach().cpu().numpy()
    normals=normal.detach().cpu().numpy()
    print(features_dc.shape,features_rest.shape)
    f_dc = features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = opacity.detach().cpu().numpy()
    scale = scaling.detach().cpu().numpy()
    rotation = rotation.detach().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(features_dc, features_rest, scaling, rotation)]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(output_file)



def save_vanilla_gaussian_ply(pth_file, ply_file):
    data,iterations= torch.load(pth_file)
    (
        active_sh_degree,
        xyz,
        features_dc,
        features_rest,
        scaling,
        rotation,
        opacity,
        t,
        scaling_t,
        velocity,
        normal,
        max_radii2D,
        xyz_gradient_accum,
        t_gradient_accum,
        denom,
        optimizer_state,
        spatial_lr_scale,
        T,
        velocity_decay,
    ) = data

    save_to_ply(xyz, normal, features_dc, features_rest, opacity, scaling, rotation, ply_file)
 
 
 # TODO: save point cloud at time t
def save_vanilla_gaussian_ply_at_time_t(pth_file, ply_file,t):
    scaling_activation = torch.exp
    scaling_inverse_activation = torch.log
    scaling_t_activation = torch.exp
    scaling_t_inverse_activation = torch.log
    opacity_activation = torch.sigmoid
    inverse_opacity_activation = inverse_sigmoid
    rotation_activation = torch.nn.functional.normalize
    normal_activation = lambda x: torch.nn.functional.normalize(x, dim=-1, eps=1e-3)

    data,iterations= torch.load(pth_file)
    (
        active_sh_degree,
        xyz,
        features_dc,
        features_rest,
        scaling,
        rotation,
        opacity,
        t,
        scaling_t,
        velocity,
        normal,
        max_radii2D,
        xyz_gradient_accum,
        t_gradient_accum,
        denom,
        optimizer_state,
        spatial_lr_scale,
        T,
        velocity_decay,
    ) = data

    save_to_ply(xyz, normal, features_dc, features_rest, opacity, scaling, rotation, ply_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert PTH file to PLY and print PLY properties.")
    parser.add_argument("pth_file", type=str, help="Path to the input PTH file.")
    parser.add_argument("ply_file", type=str, help="Path to the output PLY file.")

    args = parser.parse_args()

    save_vanilla_gaussian_ply(args.pth_file, args.ply_file)
    print_ply_properties(args.ply_file)