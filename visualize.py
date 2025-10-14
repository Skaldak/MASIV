import os
from argparse import ArgumentParser
from copy import deepcopy

import numpy as np
import open3d as o3d
import torch
import warp as wp
from diff_gauss import GaussianRasterizer as Renderer
from omegaconf import OmegaConf
from tqdm import tqdm, trange

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import Scene, DeformModel
from simulator.estimator import Estimator
from train_gs import train_gs_with_fixed_pcd
from utils.general_utils import safe_state
from utils.reg_utils import build_rotation, quat_mult
from visualizer.colormap import colormap
from visualizer.helpers import setup_camera

RENDER_MODE = "color"  # 'color', 'depth' or 'centers'
ADDITIONAL_LINES = "trajectories"  # None, 'trajectories' or 'rotations'
REMOVE_BACKGROUND = False  # False or True
FORCE_LOOP = False  # False or True

w, h = 800, 800
near, far = 0.01, 100.0
view_scale = 1.0
fps = 10
traj_frac = 50  # 1% of points
traj_length = 1
def_pix = (
    torch.tensor(np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
)
pix_ones = torch.ones(h * w, 1).cuda().float()

image_scale = 1.0


def init_camera(y_angle=0.0, center_dist=2.4, cam_height=1.3, f_ratio=0.82):
    ry = y_angle * np.pi / 180
    w2c = np.array(
        [
            [np.cos(ry), 0.0, -np.sin(ry), 0.0],
            [0.0, 1.0, 0.0, cam_height],
            [np.sin(ry), 0.0, np.cos(ry), center_dist],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    k = np.array([[f_ratio * w, 0, w / 2], [0, f_ratio * w, h / 2], [0, 0, 1]])
    return w2c, k


def load_data_deformable(dataset: ModelParams, gs_args, phys_args):
    """Load deformable gaussian data (original implementation)"""
    gaussians = GaussianModel(dataset.sh_degree, dataset.num_attribute == 10)
    scene = Scene(dataset, gaussians, load_iteration=gs_args.iteration, shuffle=False, resolution_scales=[image_scale])
    deform = DeformModel(dataset)
    deform.load_weights(dataset.model_path)

    if getattr(gs_args, "split", "train") == "train":
        views = scene.getTrainCameras(scale=image_scale)
    elif getattr(gs_args, "split", "test") == "test":
        views = scene.getTestCameras(scale=image_scale)
    else:
        raise NotImplementedError
    fids = torch.unique(torch.stack([view.fid for view in views]))
    xyz_canonical = gaussians.get_xyz.detach()
    is_fg = torch.ones_like(xyz_canonical, dtype=torch.bool)[..., 0]
    scene_data = []
    with torch.no_grad():
        for idx, fid in enumerate(tqdm(fids)):
            if getattr(phys_args, "n_frames", None) and idx >= phys_args.n_frames:
                break
            time_input = fid.unsqueeze(0).expand(1, -1)
            d_xyz, d_rotation, d_scaling = deform.step(xyz_canonical, time_input)
            rendervar = {
                "means3D": xyz_canonical + d_xyz,
                "means2D": torch.zeros_like(xyz_canonical),
                "shs": gaussians.get_features,
                "colors_precomp": None,
                "rotations": gaussians.get_rotation + d_rotation,
                "opacities": gaussians.get_opacity,
                "scales": (gaussians.get_scaling + d_scaling).repeat(1, 3),
            }
            if REMOVE_BACKGROUND:
                rendervar = {k: v[is_fg] for k, v in rendervar.items()}
            scene_data.append(rendervar)
    if REMOVE_BACKGROUND:
        is_fg = is_fg[is_fg]
    return scene_data, is_fg, views


@torch.no_grad()
def simulate(model, frames, diff=True, save_ply=False, path=None):
    """Simulate MPM trajectory using trained Estimator (exactly from predict.py)"""
    print("MPM simulation")
    print(f"Init velocity: {model.velocity.tolist()}")

    seq = []
    x0 = model.position
    x, v, C, F = None, None, None, None

    for frame_idx in trange(frames):
        if frame_idx == 0:
            x, v, C, F = model.initialize_parameters()
            if len(v) != len(x):
                v = v.repeat_interleave(x.shape[0], dim=0)
            seq.append(torch.zeros_like(x0) if diff else x0)
        else:
            x, v, C, F, *_ = model.advance(x, v, C, F)
            seq.append((x - x0) if diff else x)

    if save_ply and path:
        from utils.system_utils import write_particles

        for frame_idx in trange(frames):
            write_particles(seq[frame_idx], frame_idx, path, name="simulation")

    return seq


def load_pcd_file(file_path, iteration=None):
    """Load initial point cloud from MPM static file (exactly from predict.py)"""
    path = os.path.join(file_path, "mpm", "static_0.ply")
    pcd = o3d.io.read_point_cloud(path)
    np_pcd = np.array(pcd.points)
    vol = torch.from_numpy(np_pcd).to("cuda", dtype=torch.float32).contiguous()
    return vol


def extract_views(cameras):
    """Extract views function (exactly from predict.py)"""
    views = cameras.copy()
    fids = torch.unique(torch.stack([view.fid for view in views]))
    views = {i: [view for view in views if view.fid == fids[i]] for i in range(len(fids))}
    views = {i: sorted(v, key=lambda x: x.colmap_id) for i, v in views.items()}
    return views


def load_data_mpm(dat, opt, ppl, phy, arg, cfg):
    """Load MPM-based gaussian data - completely aligned with predict.py"""
    print("Loading MPM-based gaussian data using trained Estimator...")

    # 2. Load initial position (exactly from predict.py)
    position = load_pcd_file(dat.model_path)
    print(f"Loaded {position.shape[0]} particles from fixed PCD")

    # 3. Load initial velocity and constitutive models (exactly from predict.py)
    model = Estimator(position, cfg, dat, opt, ppl, phy)
    resume_path = model.load_weights(dat.model_path, iteration=arg.load_iter, phase="constitution")
    print(f"Constitution parameters loaded from {resume_path}")

    # 4. Get total frames (from predict.py)
    total_frames = getattr(phy, "n_frames", 30)

    # 5. Re-train appearance
    d_xyz_list = simulate(model, total_frames, diff=True, save_ply=False, path=dat.model_path)
    scene = train_gs_with_fixed_pcd(
        position,
        dat,
        opt,
        ppl,
        arg.test_iterations + list(range(10000, 40001, 1000)),
        arg.save_iterations,
        d_xyz_list[: phy.n_frames],
        phy.fps,
        force_train=arg.force_train,
        grid_size=phy.density_grid_size,
    )
    preds = simulate(model, total_frames, diff=False, save_ply=True, path=dat.model_path)

    # 6. Get views (exactly from predict.py)
    train_views = extract_views(scene.getTrainCameras(scale=dat.res_scale))
    test_views = extract_views(scene.getTestCameras(scale=dat.res_scale))
    train_frames = min(getattr(phy, "n_frames", len(d_xyz_list)), len(train_views))
    print("train frames:", train_frames, "total frames:", total_frames)

    # 7. Use the views based on split
    if getattr(arg, "split", "train") == "train":
        views = scene.getTrainCameras(scale=dat.res_scale)
    elif getattr(arg, "split", "test") == "test":
        views = scene.getTestCameras(scale=dat.res_scale)
    else:
        raise NotImplementedError

    # 8. Create scene data using MPM deformations (similar to predict.py render loop)
    xyz_canonical = scene.gaussians.get_xyz.detach()
    is_fg = torch.ones_like(xyz_canonical, dtype=torch.bool)[..., 0]
    scene_data = []

    with torch.no_grad():
        for frame_idx in tqdm(range(min(len(d_xyz_list), total_frames)), desc="Creating MPM scene data"):
            # Use the same approach as predict.py - apply d_xyz as deformation
            d_xyz = d_xyz_list[frame_idx]

            # Handle size mismatch properly (similar to predict.py logic)
            if d_xyz.shape[0] != xyz_canonical.shape[0]:
                print(f"Warning: PCD size ({d_xyz.shape[0]}) != Gaussian size ({xyz_canonical.shape[0]})")
                # Create proper mapping - extend d_xyz to match gaussian size
                if d_xyz.shape[0] < xyz_canonical.shape[0]:
                    # Pad with zeros for extra gaussians
                    pad_size = xyz_canonical.shape[0] - d_xyz.shape[0]
                    d_xyz = torch.cat([d_xyz, torch.zeros(pad_size, 3).to(d_xyz)], dim=0)
                else:
                    # Subsample if more PCD points than gaussians
                    indices = torch.randperm(d_xyz.shape[0])[: xyz_canonical.shape[0]]
                    d_xyz = d_xyz[indices]

            # Create rendervar similar to deformable version but with MPM deformations
            rendervar = {
                "means3D": xyz_canonical + d_xyz,  # Apply deformation to canonical positions
                "means2D": torch.zeros_like(xyz_canonical),
                "shs": scene.gaussians.get_features,
                "colors_precomp": None,
                "rotations": scene.gaussians.get_rotation,  # No rotation deformation from MPM
                "opacities": scene.gaussians.get_opacity,
                "scales": scene.gaussians.get_scaling.repeat(1, 3),  # No scaling deformation from MPM
            }
            if REMOVE_BACKGROUND:
                rendervar = {k: v[is_fg] for k, v in rendervar.items()}
            scene_data.append(rendervar)

    if REMOVE_BACKGROUND:
        is_fg = is_fg[is_fg]

    print(f"MPM scene data created with {len(scene_data)} frames")
    return scene_data, is_fg, views


def load_data(dat, opt, ppl, phy, arg, cfg, mode="deformable"):
    """Main data loading function with mode selection"""
    if mode == "deformable":
        return load_data_deformable(dat, arg, phy)
    elif mode == "mpm":
        return load_data_mpm(dat, opt, ppl, phy, arg, cfg)
    else:
        raise ValueError(f"Unknown visualization mode: {mode}. Choose 'deformable' or 'mpm'")


def make_lineset(all_pts, cols, num_lines):
    linesets = []
    for pts in all_pts:
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
        lineset.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(cols, np.float64))
        pt_indices = np.arange(len(lineset.points))
        line_indices = np.stack((pt_indices, pt_indices - num_lines), -1)[num_lines:]
        lineset.lines = o3d.utility.Vector2iVector(np.ascontiguousarray(line_indices, np.int32))
        linesets.append(lineset)
    return linesets


def calculate_trajectories(scene_data, is_fg):
    in_pts = [data["means3D"][is_fg][::traj_frac].contiguous().float().cpu().numpy() for data in scene_data]
    num_lines = len(in_pts[0])
    cols = np.repeat(colormap[np.arange(len(in_pts[0])) % len(colormap)][None], traj_length, 0).reshape(-1, 3)
    out_pts = []
    for t in range(len(in_pts))[traj_length:]:
        out_pts.append(np.array(in_pts[t - traj_length : t + 1]).reshape(-1, 3))
    return make_lineset(out_pts, cols, num_lines)


def calculate_rot_vec(scene_data, is_fg):
    in_pts = [data["means3D"][is_fg][::traj_frac].contiguous().float().cpu().numpy() for data in scene_data]
    in_rotation = [data["rotations"][is_fg][::traj_frac] for data in scene_data]
    num_lines = len(in_pts[0])
    cols = colormap[np.arange(num_lines) % len(colormap)]
    inv_init_q = deepcopy(in_rotation[0])
    inv_init_q[:, 1:] = -1 * inv_init_q[:, 1:]
    inv_init_q = inv_init_q / (inv_init_q**2).sum(-1)[:, None]
    init_vec = np.array([-0.1, 0, 0])
    out_pts = []
    for t in range(len(in_pts)):
        cam_rel_qs = quat_mult(in_rotation[t], inv_init_q)
        rot = build_rotation(cam_rel_qs).cpu().numpy()
        vec = (rot @ init_vec[None, :, None]).squeeze()
        out_pts.append(np.concatenate((in_pts[t] + vec, in_pts[t]), 0))
    return make_lineset(out_pts, cols, num_lines)


def render_timestep(w2c, k, timestep_data):
    with torch.no_grad():
        cam = setup_camera(w, h, k, w2c, near, far)
        im, depth, alpha, radii = Renderer(raster_settings=cam)(**timestep_data)
        return im, depth


def rgbd2pcd(im, depth, w2c, k, show_depth=False, project_to_cam_w_scale=None):
    d_near = 1.5
    d_far = 6
    invk = torch.inverse(torch.tensor(k).cuda().float())
    c2w = torch.inverse(torch.tensor(w2c).cuda().float())
    radial_depth = depth[0].reshape(-1)
    def_rays = (invk @ def_pix.T).T
    def_radial_rays = def_rays / torch.linalg.norm(def_rays, ord=2, dim=-1)[:, None]
    pts_cam = def_radial_rays * radial_depth[:, None]
    z_depth = pts_cam[:, 2]
    if project_to_cam_w_scale is not None:
        pts_cam = project_to_cam_w_scale * pts_cam / z_depth[:, None]
    pts4 = torch.concat((pts_cam, pix_ones), 1)
    pts = (c2w @ pts4.T).T[:, :3]
    if show_depth:
        cols = ((z_depth - d_near) / (d_far - d_near))[:, None].repeat(1, 3)
    else:
        cols = torch.permute(im, (1, 2, 0)).reshape(-1, 3)
    pts = o3d.utility.Vector3dVector(pts.contiguous().double().cpu().numpy())
    cols = o3d.utility.Vector3dVector(cols.contiguous().double().cpu().numpy())
    return pts, cols


if __name__ == "__main__":
    # Set up command line argument parser (aligned with predict.py)
    parser = ArgumentParser(description="Trajectory visualization")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument(
        "--trajectory",
        default="deformable",
        choices=["deformable", "mpm"],
        help="Visualization mode: 'deformable' for deformable gaussians, 'mpm' for MPM-based gaussians",
    )
    parser.add_argument("--load_iter", type=int, default=0)  # Added to match predict.py
    parser.add_argument("--force_train", type=int, default=0)  # Added to match predict.py

    mdl = ModelParams(parser)  # , sentinel=True)
    ppl = PipelineParams(parser)
    opt = OptimizationParams(parser)
    arg, phy, unk = get_combined_args(parser)

    dat = mdl.extract(arg)
    opt = opt.extract(arg)
    ppl = ppl.extract(arg)

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

    print(f"Visualization mode: {arg.trajectory}")
    scene_data, is_fg, views = load_data(dat, opt, ppl, phy, arg, cfg, mode=arg.trajectory)
    w2c, k = views[0].world_view_transform.cpu().numpy().T, views[0].intrinsic

    im, depth = render_timestep(w2c, k, scene_data[0])
    init_pts, init_cols = rgbd2pcd(im, depth, w2c, k, show_depth=(RENDER_MODE == "depth"))
    pcd = o3d.geometry.PointCloud()
    pcd.points = init_pts
    pcd.colors = init_cols

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=int(w * view_scale), height=int(h * view_scale), visible=True)
    vis.add_geometry(pcd)

    linesets = None
    lines = None
    if ADDITIONAL_LINES is not None:
        if ADDITIONAL_LINES == "trajectories":
            linesets = calculate_trajectories(scene_data, is_fg)
        elif ADDITIONAL_LINES == "rotations":
            linesets = calculate_rot_vec(scene_data, is_fg)
        lines = o3d.geometry.LineSet()
        lines.points = linesets[0].points
        lines.colors = linesets[0].colors
        lines.lines = linesets[0].lines
        vis.add_geometry(lines)

    view_k = k * view_scale
    view_k[2, 2] = 1
    view_control = vis.get_view_control()
    cparams = o3d.camera.PinholeCameraParameters()
    cparams.extrinsic = w2c
    cparams.intrinsic.intrinsic_matrix = view_k
    cparams.intrinsic.height = int(h * view_scale)
    cparams.intrinsic.width = int(w * view_scale)
    view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

    render_options = vis.get_render_option()
    render_options.point_size = view_scale
    render_options.light_on = False

    num_timesteps = len(scene_data)
    passed_frames = 0
    while True:
        if ADDITIONAL_LINES == "trajectories":
            t = int(passed_frames % (num_timesteps - traj_length)) + traj_length  # Skip t that don't have full traj.
            passed_frames = (passed_frames + 1) % (num_timesteps - traj_length)
        else:
            t = int(passed_frames % num_timesteps)
            passed_frames = (passed_frames + 1) % num_timesteps

        if FORCE_LOOP:
            num_loops = 1.4
            y_angle = 360 * t * num_loops / num_timesteps
            w2c, k = init_camera(y_angle)
            cam_params = view_control.convert_to_pinhole_camera_parameters()
            cam_params.extrinsic = w2c
            view_control.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
        else:  # Interactive control
            cam_params = view_control.convert_to_pinhole_camera_parameters()
            view_k = cam_params.intrinsic.intrinsic_matrix
            k = view_k / view_scale
            k[2, 2] = 1
            w2c = cam_params.extrinsic

        if RENDER_MODE == "centers":
            pts = o3d.utility.Vector3dVector(scene_data[t]["means3D"].contiguous().double().cpu().numpy())
            cols = o3d.utility.Vector3dVector(scene_data[t]["colors_precomp"].contiguous().double().cpu().numpy())
        else:
            im, depth = render_timestep(w2c, k, scene_data[t])
            pts, cols = rgbd2pcd(im, depth, w2c, k, show_depth=(RENDER_MODE == "depth"))
        pcd.points = pts
        pcd.colors = cols
        vis.update_geometry(pcd)

        if ADDITIONAL_LINES is not None:
            if ADDITIONAL_LINES == "trajectories":
                lt = t - traj_length
            else:
                lt = t
            lines.points = linesets[lt].points
            lines.colors = linesets[lt].colors
            lines.lines = linesets[lt].lines
            vis.update_geometry(lines)

        if not vis.poll_events():
            break
        vis.update_renderer()

    vis.destroy_window()
    del view_control
    del vis
    del render_options
