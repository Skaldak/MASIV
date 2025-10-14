import os
import shutil
import time
from argparse import ArgumentParser

import imageio
import open3d as o3d
import torch
import torch.nn.functional as F
import warp as wp
from omegaconf import OmegaConf
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points
from torch import nn
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
from gaussian_renderer import render
from scene import Scene, DeformModel
from simulator.estimator import Estimator
from train_gs import train_reconstruction, train_registration, assign_gs_to_pcd
from utils.general_utils import safe_state
from utils.system_utils import check_gs_model, write_particles, check, move

image_scale = 1.0


class Timer(object):
    """Time recorder."""

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1.0):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return "{:.1f}h".format(x / 3600)
        if x >= 60:
            return "{}m".format(round(x / 60))
        return "{}s".format(x)


def inverse_deformation(points, times, network, max_iter=1000, verbose=False):
    canonical = nn.Parameter(points.clone())
    optimizer = torch.optim.LBFGS([canonical])
    curr_loss = torch.inf

    def closure():
        nonlocal curr_loss
        optimizer.zero_grad()
        pred = canonical + network.step(canonical, times)[0]
        loss = F.mse_loss(pred, points)
        loss.backward()
        curr_loss = loss.item()
        return loss

    for i in range(max_iter):
        optimizer.step(closure)
        if verbose:
            print(curr_loss)

    return canonical.detach()


def moving_least_squares_knn(grid, origin, deform, weight=None, K=32, alpha=1.0, s=0.0, eps=1e-8):
    if isinstance(weight, torch.Tensor):
        assert weight.shape == origin.shape[:-1]
    elif weight is not None:
        weight = torch.as_tensor(weight).to(origin).expand_as(origin[..., 0])
    d_k, i_k, _ = knn_points(grid[None], origin[None], K=K, return_sorted=False)
    d_k, i_k = d_k[0], i_k[0]  # [G, K]

    origin_k = origin[i_k]  # [G, K, 3]
    deform_k = deform[i_k]  # [G, K, 3]
    weight_k = weight if weight is None else weight[i_k]  # [G, K]
    d2 = (d_k if weight_k is None else d_k / weight_k[..., None]) ** (2 * alpha)  # [G, K]

    w = 1 / (d2 + eps)  # [G, K]
    w_inv_norm = 1 / w.sum(-1)  # [G]
    origin_centroid = torch.einsum("g,gk,gkd->gd", w_inv_norm, w, origin_k)  # [G, 3]
    deform_centroid = torch.einsum("g,gk,gkd->gd", w_inv_norm, w, deform_k)  # [G, 3]
    origin_hat = origin_k - origin_centroid[:, None]  # [G, K, 3]
    deform_hat = deform_k - deform_centroid[:, None]  # [G, K, 3]

    PQt = torch.einsum("gk,gki,gkj->gij", w, origin_hat, deform_hat)  # [G, 3, 3]
    U, S, Vt = torch.linalg.svd(PQt)
    M = Vt.transpose(-1, -2) @ U.transpose(-1, -2)
    if s > 0:
        PPt = torch.einsum("gk,gki,gkj->gij", w, origin_hat, origin_hat)  # [G, 3, 3]
        rho = S.sum(-1) / PPt.diagonal(dim1=-1, dim2=-2).sum(-1)
        rho = rho.clip(min=1 - s, max=(1 / (1 - s)) if 1 - s > 0 else None)
        M = rho[..., None, None] * M
    grid_affine = torch.einsum("gij,gj->gi", M, grid - origin_centroid) + deform_centroid  # [G, 3]

    return grid_affine


def moving_least_squares(grid, origin, deform, weight=None, alpha=1.0, s=0.0, eps=1e-8):
    if isinstance(weight, torch.Tensor):
        assert weight.shape == origin.shape[:-1]
    elif weight is not None:
        weight = torch.as_tensor(weight).to(origin).expand_as(origin[..., 0])
    d = torch.norm(grid[:, None] - origin[:, :, None], dim=-1)
    d2 = (d if weight is None else d / weight[..., None]) ** (2 * alpha)

    w = 1 / (d2 + eps)  # [B, C, G]
    w_inv_norm = 1 / w.sum(1)
    origin_centroid = torch.einsum("bg,bcg,bcd->bgd", w_inv_norm, w, origin)
    deform_centroid = torch.einsum("bg,bcg,bcd->bgd", w_inv_norm, w, deform)
    origin_hat = origin[:, :, None] - origin_centroid[:, None]  # [B, C, G, 3]
    deform_hat = deform[:, :, None] - deform_centroid[:, None]  # [B, C, G, 3]

    PQt = torch.einsum("bcg,bcgi,bcgj->bgij", w, origin_hat, deform_hat)
    U, S, Vt = torch.linalg.svd(PQt)
    M = Vt.transpose(-1, -2) @ U.transpose(-1, -2)
    if s > 0:
        PPt = torch.einsum("bcg,bcgi,bcgj->bgij", w, origin_hat, origin_hat)
        rho = S.sum(-1) / PPt.diagonal(dim1=-1, dim2=-2).sum(-1)
        rho = rho.clip(min=1 - s, max=(1 / (1 - s)) if 1 - s > 0 else None)
        M = rho[..., None, None] * M
    grid_affine = torch.einsum("bgij,bgj->bgi", M, grid - origin_centroid) + deform_centroid

    return grid_affine


@torch.no_grad()
def prepare(dat, ppl, phy, load_iteration=-1):
    gaussians = GaussianModel(dat.sh_degree, dat.num_attribute == 10)
    scene = Scene(dat, gaussians, load_iteration=load_iteration, shuffle=False, resolution_scales=[image_scale])
    deform = DeformModel(dat)
    deform.load_weights(dat.model_path)

    out_path = os.path.join(dat.model_path, "dpsr")
    os.makedirs(out_path, exist_ok=True)

    bg_color = [1, 1, 1] if dat.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    views = scene.getTrainCameras(scale=image_scale)
    fids = torch.unique(torch.stack([view.fid for view in views]))
    xyz_canonical = gaussians.get_xyz.detach()
    opacity = gaussians.get_opacity.squeeze()

    grid_size = phy.density_grid_size
    density_min_th = phy.density_min_th
    density_max_th = phy.density_max_th
    num_iter = 4 if phy.random_sample else 5
    filling_grid_size = grid_size / 2**5
    opacity_threshold = phy.opacity_threshold

    gts = []

    for idx, fid in enumerate(tqdm(fids, desc="Filling progress")):
        if getattr(phy, "n_frames", None) and idx >= phy.n_frames:
            break
        time_input = fid.unsqueeze(0).expand(1, -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz_canonical, time_input)
        xyzt = xyz_canonical + d_xyz

        pcds = o3d.geometry.PointCloud()
        pcds.points = o3d.utility.Vector3dVector(xyzt[opacity > opacity_threshold].cpu().numpy())
        o3d.io.write_point_cloud(os.path.join(out_path, f"pointcloud_init_{idx:02d}.ply"), pcds)

        bbox_mins = xyzt[opacity > opacity_threshold].min(dim=0)[0] - grid_size
        bbox_maxs = xyzt[opacity > opacity_threshold].max(dim=0)[0] + grid_size
        bbox_bounds = bbox_maxs - bbox_mins
        volume_size = torch.round(bbox_bounds / filling_grid_size).to(torch.int64) + 1
        grid_ids = [torch.arange(size) for size in volume_size]
        grid_coords = torch.stack(torch.meshgrid(*grid_ids, indexing="ij"), dim=-1).reshape(-1, 3) * filling_grid_size
        grid_coords = grid_coords.to(xyzt)
        init_inner_points = grid_coords + bbox_mins.reshape(1, 3)
        curr_views = [view for view in views if view.fid == fid]
        for viewpoint_cam in tqdm(curr_views, desc="Rendering progress"):
            results = render(
                viewpoint_cam,
                gaussians,
                ppl,
                background,
                d_xyz=d_xyz,
                d_rotation=d_rotation,
                d_scaling=d_scaling,
            )
            depth = results["depth"][0]
            render_mask = torch.logical_and(results["render"].sum(0) != 0, viewpoint_cam.original_image.sum(0) != 0)
            pix_w, pix_h, pix_d = viewpoint_cam.pw2pix(init_inner_points)
            # remove points that are outside the image space
            in_mask = viewpoint_cam.is_in_view(pix_w, pix_h)
            init_inner_points = init_inner_points[in_mask]
            pix_w, pix_h, pix_d = pix_w[in_mask], pix_h[in_mask], pix_d[in_mask]
            # remove points that the projected pixels are outside the object mask
            pix_mask = render_mask[pix_h, pix_w]
            init_inner_points = init_inner_points[pix_mask]
            pix_w, pix_h, pix_d = pix_w[pix_mask], pix_h[pix_mask], pix_d[pix_mask]
            # remove points whose depth values are smaller than those from depth map
            render_pix_d = depth[pix_h, pix_w]
            depth_mask = render_pix_d < pix_d
            init_inner_points = init_inner_points[depth_mask]
            # remove outliers in xyzt
            render_mask = results["render"].sum(0) > 1 / 255
            pix_w, pix_h, pix_d = viewpoint_cam.pw2pix(xyzt)
            in_mask = viewpoint_cam.is_in_view(pix_w, pix_h)
            pix_w, pix_h, pix_d = pix_w[in_mask], pix_h[in_mask], pix_d[in_mask]
            xyzt = xyzt[in_mask]
            pix_mask = render_mask[pix_h, pix_w]
            xyzt = xyzt[pix_mask]
        curr_grid_size = grid_size / 2
        volume_size = torch.round(bbox_bounds / curr_grid_size).to(torch.int64) + 1
        bbox_maxs = bbox_mins + (volume_size - 1) * curr_grid_size
        bbox_bounds = bbox_maxs - bbox_mins
        density_volume = torch.zeros(volume_size.cpu().numpy().tolist()).to(init_inner_points)
        ids = torch.round((init_inner_points - bbox_mins.reshape(1, 3)) / curr_grid_size).to(torch.int64)
        density_volume[ids.T[0], ids.T[1], ids.T[2]] = 1.0
        ids = torch.round((xyzt - bbox_mins.reshape(1, 3)) / curr_grid_size).to(torch.int64)
        density_volume[ids.T[0], ids.T[1], ids.T[2]] = 1.0
        weight = torch.ones((1, 1, 3, 3, 3)).to(xyzt)
        weight = weight / weight.sum()
        for i in range(2, num_iter):
            curr_grid_size = grid_size / 2**i
            volume_size = torch.round(bbox_bounds / curr_grid_size).to(torch.int64) + 1
            bbox_maxs = bbox_mins + (volume_size - 1) * curr_grid_size
            grid_xyz = (
                torch.stack(
                    torch.meshgrid(
                        torch.linspace(0, volume_size[0] - 1, volume_size[0]),
                        torch.linspace(0, volume_size[1] - 1, volume_size[1]),
                        torch.linspace(0, volume_size[2] - 1, volume_size[2]),
                    ),
                    dim=-1,
                ).to(bbox_mins)
                * curr_grid_size
                + bbox_mins[None, None, None]
            )
            ids_norm = (grid_xyz - bbox_mins[None, None, None]) / bbox_bounds[None, None, None] * 2 - 1
            ids_norm = ids_norm[None].flip((-1,))
            density_volume = torch.nn.functional.grid_sample(
                density_volume[None, None], ids_norm, mode="bilinear", align_corners=True
            )
            density_volume = torch.nn.functional.conv3d(density_volume, weight=weight, padding="same")[0, 0]
            density_volume[density_volume < 0.5] = 0.0
            ids = torch.round((init_inner_points - bbox_mins.reshape(1, 3)) / curr_grid_size).to(torch.int64)
            density_volume[ids.T[0], ids.T[1], ids.T[2]] = 1.0
            ids = torch.round((xyzt - bbox_mins.reshape(1, 3)) / curr_grid_size).to(torch.int64)
            density_volume[ids.T[0], ids.T[1], ids.T[2]] = 1.0
            bbox_bounds = bbox_maxs - bbox_mins
        for i in range(20):
            density_volume = torch.nn.functional.conv3d(density_volume[None, None], weight=weight, padding="same")[0, 0]
            density_volume[density_volume < 0.5] = 0.0
            ids = torch.round((init_inner_points - bbox_mins.reshape(1, 3)) / curr_grid_size).to(torch.int64)
            density_volume[ids.T[0], ids.T[1], ids.T[2]] = 1.0
            ids = torch.round((xyzt - bbox_mins.reshape(1, 3)) / curr_grid_size).to(torch.int64)
            density_volume[ids.T[0], ids.T[1], ids.T[2]] = 1.0
        if phy.random_sample:
            density_volume = torch.nn.functional.conv3d(density_volume[None, None], weight=weight, padding="same")[0, 0]
            half_grid_xyz = (
                torch.stack(
                    torch.meshgrid(
                        torch.linspace(0, volume_size[0] - 0.5, 2 * (volume_size[0] - 1)),
                        torch.linspace(0, volume_size[1] - 0.5, 2 * (volume_size[1] - 1)),
                        torch.linspace(0, volume_size[2] - 0.5, 2 * (volume_size[2] - 1)),
                    ),
                    -1,
                ).to(bbox_mins)
                * curr_grid_size
                + bbox_mins[None, None, None]
            )
            ids_norm = (half_grid_xyz - bbox_mins[None, None, None]) / bbox_bounds[None, None, None] * 2 - 1
            ids_norm = ids_norm[None].flip((-1,))
            density_half_grid_xyz = torch.nn.functional.grid_sample(
                density_volume[None, None], ids_norm, mode="bilinear", align_corners=True
            )[0, 0]
            half_grid_xyz = half_grid_xyz[density_half_grid_xyz > 0.5]
            delta = (torch.rand_like(half_grid_xyz) * curr_grid_size * 0.5).to(xyzt)
            particles = half_grid_xyz + delta
            ids_norm = (particles[None, None] - bbox_mins[None, None, None]) / bbox_bounds[None, None, None] * 2 - 1
            ids_norm = ids_norm[None].flip((-1,))
            density_particles = torch.nn.functional.grid_sample(
                density_volume[None, None], ids_norm, mode="bilinear", align_corners=True
            )[0, 0, 0, 0]
            sampled_pts = particles[density_particles > density_min_th]
            curr_grid_size = curr_grid_size / 2
            gts.append(sampled_pts)
            write_particles(sampled_pts, idx, dat.model_path, "dynamic")
            if fid == 0.0:
                vol = sampled_pts
                vol_opacity = density_particles[density_particles > density_min_th]
                vol_surface_mask = density_particles[density_particles > density_min_th] < density_max_th
                vol_surface = torch.arange(
                    vol_surface_mask.shape[0],
                    dtype=torch.int64,
                    device=vol.device,
                )[vol_surface_mask]
                write_particles(vol, 0, dat.model_path, "static")
        else:
            density_volume = torch.nn.functional.conv3d(density_volume[None, None], weight=weight, padding="same")[0, 0]
            internal_mask = density_volume >= density_min_th
            sampled_pts = torch.stack(torch.where(internal_mask), dim=-1) * curr_grid_size + bbox_mins.reshape(1, 3)
            density_volume_smoothed = torch.nn.functional.conv3d(
                density_volume[None, None], weight=weight, padding="same"
            )[0, 0]
            surface_mask = (density_volume_smoothed > 0) * (density_volume_smoothed < density_max_th) * internal_mask
            gts.append(sampled_pts)
            write_particles(sampled_pts, idx, dat.model_path, "dynamic")
            if fid == 0.0:
                vol = sampled_pts
                vol_opacity = density_volume[internal_mask]
                vol_surface_mask = surface_mask[internal_mask]
                vol_surface = torch.arange(
                    vol_surface_mask.shape[0],
                    dtype=torch.int64,
                    device=vol.device,
                )[vol_surface_mask]
                write_particles(vol, 0, dat.model_path, "static")

    train_cams, test_cams, cameras_extent = scene.overwrite_alphas(ppl, dat, deform)
    cam_info = {
        "train_cams": train_cams,
        "test_cams": test_cams,
        "cameras_extent": cameras_extent,
    }

    return gts, vol, vol_opacity, vol_surface, deform, cam_info


def train_deformation(model, data, iterations=10000):
    best_loss = float("inf")
    best_iter = -1
    timer = Timer()

    best_path = os.path.join(dat.model_path, "deformation/iteration_-1")
    best_file = os.path.join(best_path, "deformation.pth")
    os.makedirs(best_path, exist_ok=True)

    xyz_canonical = nn.Parameter(data["canonical"])
    lr = model.optimizer.param_groups[0]["lr"]
    params = [
        {"params": [xyz_canonical], "lr": lr, "name": "canonical"},
        {"params": deform.deform.parameters(), "lr": lr, "name": "deform"},
    ]
    model.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
    model.deform.train()

    for iteration in range(1, iterations + 1):
        loss = 0.0
        for frame in range(len(data["gts"])):
            gt = data["gts"][frame]
            time_input = data["train_views"][frame][0].fid[None]
            d_xyz, *_ = model.step(xyz_canonical, time_input)
            pred = xyz_canonical + d_xyz
            loss += chamfer_distance(pred[None], gt[None])[0]
        loss.backward()
        with torch.no_grad():
            if loss < best_loss:
                best_loss, best_iter = loss.item(), iteration
                state_dict = model.deform.state_dict()
                state_dict["canonical"] = xyz_canonical.detach()
                torch.save(state_dict, best_file)
            if iteration == iterations:
                state_dict = model.deform.state_dict()
                state_dict["canonical"] = xyz_canonical.detach()
                last_path = os.path.join(dat.model_path, f"deformation/iteration_{iteration}")
                last_file = os.path.join(last_path, "deformation.pth")
                os.makedirs(last_path, exist_ok=True)
                torch.save(state_dict, last_file)
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        model.update_learning_rate(iteration)
        if iteration % 100 == 0:
            messages = [
                f"[{os.path.basename(dat.model_path)} deformation {iteration}/{iterations}]",
                f"l: {loss.item():.4e}",
                f"lr: {model.optimizer.param_groups[0]['lr']:.4e}",
                f"elp: {timer.measure()}",
                f"est: {timer.measure(iteration / iterations)}",
            ]
            print(" | ".join(messages))

    model.deform.eval()
    best_state_dict = torch.load(best_file, weights_only=True, map_location="cpu")
    model.deform.load_state_dict(best_state_dict, strict=False)
    xyz_canonical = best_state_dict["canonical"].to(xyz_canonical)
    print(f"best iteration {best_iter} loaded")

    data.update(canonical=xyz_canonical.detach(), deform=model)


def train_simulation(model, data, phase, max_frames=-1, iterations=100):
    model.set_phase(phase, max_frames)

    best_loss = float("inf")
    best_iter = -1
    timer = Timer()

    for iteration in range(1, iterations + 1):
        loss, loss_geo, loss_pho, loss_traj, loss_velo = model.forward_loss(data, mode="train")
        with torch.no_grad():
            model.save_weights(dat.model_path, iteration=iteration)
            if loss < best_loss:
                best_loss, best_iter = loss.item(), iteration
                model.save_weights(dat.model_path, iteration=-1)
        loss.backward()
        grad_norm_messages = model.optimizer.step()
        model.optimizer.update_learning_rate()
        model.optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        with torch.inference_mode():
            messages = [
                f"[{os.path.basename(dat.model_path)} simulation {iteration}/{iterations}]",
                f"l: {loss.item():.4e}",
                f"l_geo: {loss_geo:.4e}",
                f"l_pho: {loss_pho:.4e}",
                f"l_traj: {loss_traj:.4e}",
                f"l_velo: {loss_velo:.4e}",
            ]
            if phase == "velocity":
                velocity = [model.velocity.min(0).values.cpu().tolist(), model.velocity.max(0).values.cpu().tolist()]
                messages += [
                    f"v-lr: {model.optimizer.velocity_optimizer.param_groups[0]['lr']:.4e}",
                    f"v-range: {[[round(x, 4) for x in v] for v in velocity]}",
                ]
            elif phase == "constitution":
                messages += [
                    f"e-lr: {model.optimizer.elasticity_optimizer.param_groups[0]['lr']:.4e}",
                    grad_norm_messages[0],
                    f"p-lr: {model.optimizer.plasticity_optimizer.param_groups[0]['lr']:.4e}",
                    grad_norm_messages[1],
                ]
            messages += [f"elp: {timer.measure()}", f"est: {timer.measure(iteration / iterations)}"]
            print(" | ".join(messages))
            if iteration % 10 == 0:
                save_path = os.path.join(dat.model_path, f"{model.optimizer.phase}/iteration_{iteration}")
                os.makedirs(save_path, exist_ok=True)
                frames = model.forward_render(data, mode="test")
                for k, v in frames.items():
                    imageio.mimwrite(os.path.join(save_path, f"view_{k}.mp4"), v)

    model.load_weights(dat.model_path, iteration=-1, phase=phase)
    print(f"best iteration {best_iter} loaded")


def extract_views(cameras):
    views = cameras[image_scale].copy()
    fids = torch.unique(torch.stack([view.fid for view in views]))
    views = {i: [view for view in views if view.fid == fids[i]] for i in range(len(fids))}
    views = {i: sorted(v, key=lambda x: x.colmap_id) for i, v in views.items()}
    return views


if __name__ == "__main__":
    start_time = time.time()

    # Set up command line argument parser
    parser = ArgumentParser(description="Physical parameter estimation")
    parser.add_argument("--overwrite", action="store_true", default=False)
    mdl = ModelParams(parser)  # , sentinel=True)
    ppl = PipelineParams(parser)
    opt = OptimizationParams(parser)
    arg, phy, unk = get_combined_args(parser)

    dat = mdl.extract(arg)
    opt = opt.extract(arg)
    ppl = ppl.extract(arg)
    opt.warm_up_frames = phy.vel_estimation_frames

    cfg = OmegaConf.load("config/default.yaml")
    cfg.merge_with(unk)

    print("Config:", arg.config_path)
    print("Data:", arg.source_path)
    print("Output:", arg.model_path)
    print(OmegaConf.to_yaml(cfg))

    # Initialize system state (RNG)
    safe_state(arg.quiet)

    wp.init()
    wp.config.verify_cuda = True
    wp.ScopedTimer.enabled = False
    wp.set_module_options({"fast_math": False})

    # 1. train def gs
    test_iterations = arg.test_iterations + list(range(10000, 40001, 1000))
    if "real_capture" in dat.source_path:
        if not check_gs_model(dat.model_path, arg.save_iterations, postfix="_static"):
            dat.mode = "static"
            train_reconstruction(dat, opt, ppl, test_iterations, arg.save_iterations)
            move(os.path.join(arg.model_path, "gs"), os.path.join(arg.model_path, "gs_static"))
            move(os.path.join(arg.model_path, "img"), os.path.join(arg.model_path, "img_static"))
            move(os.path.join(arg.model_path, "deform"), os.path.join(arg.model_path, "deform_static"))
            move(os.path.join(arg.model_path, "point_cloud"), os.path.join(arg.model_path, "point_cloud_static"))
            torch.cuda.empty_cache()
        if not check_gs_model(dat.model_path, [cfg.registration.iterations], postfix="_regist"):
            train_registration(dat, ppl, cfg, load_iteration=arg.save_iterations[-1])
            shutil.copytree(
                os.path.join(arg.model_path, "point_cloud_regist"),
                os.path.join(arg.model_path, "point_cloud"),
                dirs_exist_ok=True,
            )
            torch.cuda.empty_cache()
        if not check_gs_model(dat.model_path, arg.save_iterations):
            dat.mode = "dynamic"
            train_reconstruction(
                dat, opt, ppl, test_iterations, arg.save_iterations, load_iteration=cfg.registration.iterations
            )
            torch.cuda.empty_cache()
    elif not check_gs_model(dat.model_path, arg.save_iterations):
        dat.mode = "dynamic"
        train_reconstruction(dat, opt, ppl, test_iterations, arg.save_iterations)
        torch.cuda.empty_cache()
    gts, position, opacity, surface, deform, cam_info = prepare(dat, ppl, phy, load_iteration=arg.iteration)
    torch.cuda.empty_cache()

    # 2. train def
    train_views = extract_views(cam_info["train_cams"])
    test_views = extract_views(cam_info["test_cams"])
    data = dict(canonical=position.clone(), gts=gts, train_views=train_views, test_views=test_views)
    if not check(dat.model_path, "deformation", cfg.deformation.num_epochs):
        deform.train_setting(opt)
        # data["canonical"] = inverse_deformation(data["canonical"], torch.zeros([1, 1]).to(data["canonical"]), deform)
        train_deformation(deform, data, iterations=cfg.deformation.num_epochs)
    else:
        best_path = os.path.join(dat.model_path, "deformation/iteration_-1/deformation.pth")
        state_dict = torch.load(best_path, weights_only=True, map_location="cpu")
        deform.deform.load_state_dict(state_dict, strict=False)
        # data["canonical"] = state_dict["canonical"].to(position)
        data.update(deform=deform)
        print("deformation parameters loaded")
    torch.cuda.empty_cache()

    # 3. data preparation
    scene = assign_gs_to_pcd(
        position,
        opacity,
        dat,
        cam_info,
        phy.density_grid_size,
    )
    phy.n_frames = phy.get("n_frames", len(gts))
    model = Estimator(position, cfg, dat, opt, ppl, phy)
    bg_color = [1, 1, 1] if dat.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=model.device)
    data.update(background=background, scene=scene)
    torch.cuda.empty_cache()

    # 4. estimate velocity
    if arg.overwrite or not check(dat.model_path, "velocity", cfg.velocity.num_epochs):
        if cfg.velocity.resume is not None:
            resume_path = model.load_weights(dat.model_path, iteration=cfg.velocity.resume, phase="velocity")
            print(f"velocity parameters loaded from {resume_path}")
        train_simulation(
            model, data, "velocity", max_frames=phy.vel_estimation_frames, iterations=cfg.velocity.num_epochs
        )
    else:
        resume_path = model.load_weights(dat.model_path, iteration=-1, phase="velocity")
        print(f"velocity parameters loaded from {resume_path}")
    torch.cuda.empty_cache()

    # 5. estimate physical parameters
    if arg.overwrite or not check(dat.model_path, "constitution", cfg.meta.num_epochs):
        if cfg.meta.resume is not None:
            resume_path = model.load_weights(dat.model_path, iteration=cfg.meta.resume, phase="constitution")
            print(f"constitution parameters loaded from {resume_path}")
        train_simulation(model, data, "constitution", max_frames=len(gts), iterations=cfg.meta.num_epochs)
    else:
        resume_path = model.load_weights(dat.model_path, iteration=-1, phase="constitution")
        print(f"constitution parameters loaded from {resume_path}")
    torch.cuda.empty_cache()

    print("consume time {}".format(time.time() - start_time))
