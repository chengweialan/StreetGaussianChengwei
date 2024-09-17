#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import cv2
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.dynamic_model import scale_grads
from scene.cameras import Camera
from utils.sh_utils import eval_sh
import torch.nn.functional as F
import numpy as np
import os
import kornia
from utils.loss_utils import psnr, ssim, tv_loss,better_ssim
# from render import render_all
EPS = 1e-5

def render(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
           override_color=None, env_map=None,
           time_shift=None, other=[], mask=None, is_training=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color if env_map is not None else torch.zeros(3, device="cuda"),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = None
    rotations = None
    cov3D_precomp = None

    if time_shift is not None:
        means3D = pc.get_xyz_SHM(viewpoint_camera.timestamp-time_shift)
        means3D = means3D + pc.get_inst_velocity * time_shift
        marginal_t = pc.get_marginal_t(viewpoint_camera.timestamp-time_shift)
    else:
        means3D = pc.get_xyz_SHM(viewpoint_camera.timestamp)
        marginal_t = pc.get_marginal_t(viewpoint_camera.timestamp)
    opacity = opacity * marginal_t

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, pc.get_max_sh_channels)
            dir_pp = (means3D.detach() - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)).detach()
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    feature_list = other
    normals = pc.get_normal(viewpoint_camera.c2w, means3D, from_scaling=False)
    # Transform normals to camera space
    normals = (normals @ viewpoint_camera.world_view_transform[:3, :3])

    if len(feature_list) > 0:
        features = torch.cat(feature_list, dim=1)
        S_other = features.shape[1]
    else:
        features = torch.zeros_like(means3D[:, :0])
        S_other = 0
    
    # Prefilter
    if mask is None:
        mask = marginal_t[:, 0] > 0.05
    else:
        mask = mask & (marginal_t[:, 0] > 0.05)
    masked_means3D = means3D[mask]
    masked_xyz_homo = torch.cat([masked_means3D, torch.ones_like(masked_means3D[:, :1])], dim=1)
    masked_depth = (masked_xyz_homo @ viewpoint_camera.world_view_transform[:, 2:3])
    depth_alpha = torch.zeros(means3D.shape[0], 2, dtype=torch.float32, device=means3D.device)
    depth_alpha[mask] = torch.cat([
        masked_depth,
        torch.ones_like(masked_depth)
    ], dim=1)
    features = torch.cat([features, depth_alpha, normals], dim=1)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    contrib, rendered_image, rendered_feature, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        features = features,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        mask = mask)
    
    rendered_other, rendered_depth, rendered_opacity, rendered_normal = rendered_feature.split([S_other, 1, 1, 3], dim=0)
    # rendered_normal = F.normalize(rendered_normal, dim=0)
    rendered_normal = rendered_normal * 0.5 + 0.5 # [-1, 1] -> [0, 1]
    rendered_image_before = rendered_image
    if env_map is not None:
        bg_color_from_envmap = env_map(viewpoint_camera.get_world_directions(is_training).permute(1, 2, 0)).permute(2, 0, 1)
        rendered_image = rendered_image + (1 - rendered_opacity) * bg_color_from_envmap

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "render_nobg": rendered_image_before,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "contrib": contrib,
            "depth": rendered_depth,
            "alpha": rendered_opacity,
            "normal": rendered_normal,
            "feature": rendered_other}

idx=0
def calculate_loss(gaussians : GaussianModel, viewpoint_camera : Camera, args, render_pkg : dict, env_map, iteration, camera_id):

    global idx
    
    log_dict = {}

    
    image = render_pkg["render"]
    depth = render_pkg["depth"]
    alpha = render_pkg["alpha"]
    log_dict = {}

    feature = render_pkg['feature'] / alpha.clamp_min(EPS)
    t_map = feature[0:1]
    v_map = feature[1:]
    
    sky_mask = viewpoint_camera.sky_mask.cuda() if viewpoint_camera.sky_mask is not None else torch.zeros_like(alpha, dtype=torch.bool)

    dynamic_mask=viewpoint_camera.dynamic_mask.cuda() if viewpoint_camera.dynamic_mask is not None else alpha



    sky_depth = 900
    depth = depth / alpha.clamp_min(EPS)
    if env_map is not None:
        if args.depth_blend_mode == 0:  # harmonic mean
            depth = 1 / (alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth).clamp_min(EPS)
        elif args.depth_blend_mode == 1:
            depth = alpha * depth + (1 - alpha) * sky_depth
        
    gt_image = viewpoint_camera.original_image.cuda()
    
    loss_l1 = F.l1_loss(image, gt_image, reduction='none') # [3, H, W]
    loss_ssim = 1.0 - ssim(image, gt_image, size_average=False) # [3, H, W]

    log_dict['loss_l1'] = loss_l1.mean().item()
    log_dict['loss_ssim'] = loss_ssim.mean().item()
    uncertainty_loss = 0
    metrics = {}
    loss_mult = torch.ones_like(depth, dtype=depth.dtype)
    if gaussians.uncertainty_model is not None:
        del loss_mult

        sky_alpha=sky_mask if iteration < args.alpha_sky_start else (1-alpha) # use ~alpha as sky_mask
        
        uncertainty_loss, metrics, loss_mult = gaussians.uncertainty_model.get_loss(gt_image, image.detach(), sky_mask=sky_alpha,_cache_entry=('train', camera_id))

        if args.uncertainty_stage=="stage2":
            with torch.no_grad():
                loss_mult = (loss_mult > 1).to(dtype=loss_mult.dtype)
        else:
            loss_mult = (loss_mult > 1).to(dtype=loss_mult.dtype) # [1, H, W]


        if args.uncertainty_stage == "stage1":

            if iteration < args.uncertainty_warmup_start:
                loss_mult = torch.ones_like(depth, dtype=depth.dtype)
            elif iteration < args.uncertainty_warmup_start + args.uncertainty_warmup_iters:
                p = (iteration - args.uncertainty_warmup_start) / args.uncertainty_warmup_iters
                loss_mult = 1 + p * (loss_mult - 1)
            if args.uncertainty_center_mult: # default: False
                loss_mult = loss_mult.sub(loss_mult.mean() - 1).clamp(0, 2)
            if args.uncertainty_scale_grad: # default: False
                image = scale_grads(image, loss_mult)
                loss_mult = torch.ones_like(depth, dtype=depth.dtype)
        else:
            if iteration < args.dynamic_mask_epoch:
                loss_mult = torch.ones_like(depth, dtype=depth.dtype)


    # loss_mult = torch.ones_like(depth, dtype=depth.dtype)
    # loss_mult[dynamic_mask]=0

    # Saving

    save_dir=args.visualize_dir

    if idx%100==0:
        loss_mult_num=loss_mult.cpu().detach().numpy()
        image_np = image.detach().cpu().numpy().transpose(1,2,0)*255.0
        gt_np=gt_image.detach().cpu().numpy().transpose(1,2,0)*255.0
        cv2.imwrite(os.path.join(save_dir, 'vis',f'{idx:06d}_image.png'),cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(save_dir, 'vis',f'{idx:06d}_gt.png'),cv2.cvtColor(gt_np, cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(save_dir, 'vis',f'{idx:06d}_loss_mult.png'),loss_mult_num.squeeze()*255.0)
    

    # Detach uncertainty loss if in protected iter after opacity reset
    idx+=1
    if args.uncertainty_stage == "stage1":
        last_densify_iter = min(iteration, args.densify_until_iter - 1)
        last_dentify_iter = (last_densify_iter // args.opacity_reset_interval) * args.opacity_reset_interval
        if iteration < last_dentify_iter + args.uncertainty_protected_iters:
            # Keep track of max radii in image-space for pruning
            try:
                uncertainty_loss = uncertainty_loss.detach()  # type: ignore
            except AttributeError:
                pass

        loss = (1.0 - args.lambda_dssim) * (loss_l1 * loss_mult).mean() \
            + args.lambda_dssim * (loss_ssim * loss_mult).mean() \
                + uncertainty_loss
    else:
        loss = (1.0 - args.lambda_dssim) * loss_l1.mean() \
            + args.lambda_dssim * loss_ssim.mean()   

    psnr_for_log = psnr(image, gt_image).double()
    log_dict["psnr"] = psnr_for_log

    if args.lambda_lidar > 0:
        assert viewpoint_camera.pts_depth is not None
        pts_depth = viewpoint_camera.pts_depth.cuda()

        mask = (pts_depth > 0) & (loss_mult > 1 - EPS)
        loss_lidar =  torch.abs(1 / (pts_depth[mask] + 1e-5) - 1 / (depth[mask] + 1e-5)).mean()
        if args.lidar_decay > 0:
            iter_decay = np.exp(-iteration / 8000 * args.lidar_decay)
        else:
            iter_decay = 1
        log_dict['loss_lidar'] = loss_lidar.item()
        loss += iter_decay * args.lambda_lidar * loss_lidar

    if args.lambda_normal > 0 and args.load_normal_map:
        alpha_mask = (alpha.data > EPS).repeat(3, 1, 1) # (3, H, W) detached       
        rendered_normal = render_pkg['normal'] # (3, H, W)
        gt_normal = viewpoint_camera.normal_map.cuda()
        loss_normal = F.l1_loss(rendered_normal[alpha_mask], gt_normal[alpha_mask])
        loss_normal += tv_loss(rendered_normal)
        log_dict['loss_normal'] = loss_normal.item()
        loss += args.lambda_normal * loss_normal

    if args.lambda_t_reg > 0 and args.enable_dynamic:
        if iteration>args.dynamic_mask_epoch:
            loss_t_reg=-(torch.abs(t_map)*loss_mult).mean()*5.0
        else:
            loss_t_reg = -torch.abs(t_map).mean()
        log_dict['loss_t_reg'] = loss_t_reg.item()
        loss += args.lambda_t_reg * loss_t_reg

    # if args.lambda_v_reg > 0 and args.enable_dynamic:
    #     loss_v_reg = torch.abs(v_map).mean()
    #     log_dict['loss_v_reg'] = loss_v_reg.item()
    #     loss += args.lambda_v_reg * loss_v_reg

    if args.lambda_v_reg > 0 and args.enable_dynamic:
        if iteration > args.dynamic_mask_epoch:
            loss_v_reg = (torch.abs(v_map) * loss_mult).mean() * 10.0
        else:
            loss_v_reg = (torch.abs(v_map)).mean()
        log_dict['loss_v_reg'] = loss_v_reg.item()
        loss += args.lambda_v_reg * loss_v_reg

    if args.lambda_inv_depth > 0:
        inverse_depth = 1 / (depth + 1e-5)
        loss_inv_depth = kornia.losses.inverse_depth_smoothness_loss(inverse_depth[None], gt_image[None])
        log_dict['loss_inv_depth'] = loss_inv_depth.item()
        loss = loss + args.lambda_inv_depth * loss_inv_depth

    if args.lambda_v_smooth > 0 and args.enable_dynamic:
        loss_v_smooth = kornia.losses.inverse_depth_smoothness_loss(v_map[None], gt_image[None])
        log_dict['loss_v_smooth'] = loss_v_smooth.item()
        loss = loss + args.lambda_v_smooth * loss_v_smooth

    if args.lambda_t_smooth > 0 and args.enable_dynamic:
        loss_t_smooth = kornia.losses.inverse_depth_smoothness_loss(t_map[None], gt_image[None])
        log_dict['loss_t_smooth'] = loss_t_smooth.item()
        loss = loss + args.lambda_t_smooth * loss_t_smooth
    
    if args.lambda_sky_opa > 0:
        o = alpha.clamp(1e-6, 1-1e-6)
        sky = sky_mask.float()
        loss_sky_opa = (-sky * torch.log(1 - o)).mean()
        log_dict['loss_sky_opa'] = loss_sky_opa.item()
        loss = loss + args.lambda_sky_opa * loss_sky_opa

    if args.lambda_opacity_entropy > 0:
        o = alpha.clamp(1e-6, 1 - 1e-6)
        loss_opacity_entropy =  -(o*torch.log(o)).mean()
        log_dict['loss_opacity_entropy'] = loss_opacity_entropy.item()
        loss = loss + args.lambda_opacity_entropy * loss_opacity_entropy

    extra_render_pkg = {}
    extra_render_pkg['t_map'] = t_map
    extra_render_pkg['v_map'] = v_map
    extra_render_pkg['depth'] = depth
    extra_render_pkg['dynamic_mask'] = loss_mult 
    
    log_dict.update(metrics)
    
    return loss, log_dict, extra_render_pkg

def render_wrapper(args, viewpoint_camera : Camera, gaussians : GaussianModel, background : torch.Tensor, time_interval : float, env_map, iterations, camera_id):

    
    # render v and t scale map
    v = gaussians.get_inst_velocity
    t_scale = gaussians.get_scaling_t.clamp_max(2)
    other = [t_scale, v]

    if np.random.random() < args.lambda_self_supervision:
        time_shift = 3*(np.random.random() - 0.5) * time_interval
    else:
        time_shift = None

    render_pkg = render(viewpoint_camera, gaussians, args, background, env_map=env_map, other=other, time_shift=time_shift, is_training=True)
    
    loss, log_dict, extra_render_pkg = calculate_loss(gaussians, viewpoint_camera, args, render_pkg, env_map, iterations, camera_id)
    
    render_pkg.update(extra_render_pkg)
    
    return loss, log_dict, render_pkg

