import os
import random
import subprocess
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import open3d as o3d
import scipy
import torch
import torchvision
import warp as wp
from omegaconf import OmegaConf
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm, trange

from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import render
from simulator.estimator import Estimator
from train_gs import train_gs_with_fixed_pcd
from utils.general_utils import safe_state
from utils.system_utils import write_particles

image_scale = 1.0


@torch.no_grad()
def simulate(model, frames, diff=True, save_ply=False, path=None):
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
        for frame_idx in trange(frames):
            write_particles(seq[frame_idx], frame_idx, path, name="simulation")

    return seq


def discretize(pcd, vs):
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd.cpu().numpy())
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_o3d, voxel_size=vs)
    voxels = voxel_grid.get_voxels()
    np_pcd = np.zeros((len(voxels), 3))
    for j in range(len(voxels)):
        c = voxel_grid.get_voxel_center_coordinate(voxels[j].grid_index)
        np_pcd[j, :] = c
    return torch.from_numpy(np_pcd).to("cuda", dtype=torch.float32).contiguous()


def load_pcd_file(file_path, iteration=None):
    # path = os.path.join(file_path, 'point_cloud_fix_pcd', f'iteration_{iteration}', 'point_cloud.ply')
    path = os.path.join(file_path, "mpm", "static_0.ply")
    pcd = o3d.io.read_point_cloud(path)
    np_pcd = np.array(pcd.points)
    vol = torch.from_numpy(np_pcd).to("cuda", dtype=torch.float32).contiguous()
    return vol


def load_gt_pcds(path):
    gts = []
    indices = [ply.split(".")[0] for ply in os.listdir(path)]
    indices.sort()
    n_digits = len(indices[0])
    indices = [int(idx) for idx in indices]
    indices.sort()
    for idx in range(len(indices)):
        pcd = o3d.io.read_point_cloud(os.path.join(path, f"{idx:0{n_digits}d}.ply"))
        np_pcd = np.array(pcd.points)
        gts.append(torch.from_numpy(np_pcd).to("cuda", dtype=torch.float32).contiguous())
    return gts


def emd_func(x, y, pkg="torch"):
    if pkg == "numpy":
        # numpy implementation
        x_ = np.repeat(np.expand_dims(x, axis=1), y.shape[0], axis=1)  # x: [N, M, D]
        y_ = np.repeat(np.expand_dims(y, axis=0), x.shape[0], axis=0)  # y: [N, M, D]
        cost_matrix = np.linalg.norm(x_ - y_, 2, axis=2)
        try:
            ind1, ind2 = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=False)
        except:
            # pdb.set_trace()
            print("Error in linear sum assignment!")

        emd = np.mean(np.linalg.norm(x[ind1] - y[ind2], 2, axis=1))
    else:
        # torch implementation
        x_ = x[:, None, :].repeat(1, y.size(0), 1)  # x: [N, M, D]
        y_ = y[None, :, :].repeat(x.size(0), 1, 1)  # y: [N, M, D]
        dis = torch.norm(torch.add(x_, -y_), 2, dim=2)  # dis: [N, M]
        cost_matrix = dis.detach().cpu().numpy()
        try:
            ind1, ind2 = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=False)
        except:
            # pdb.set_trace()
            print("Error in linear sum assignment!")

        emd = torch.mean(torch.norm(torch.add(x[ind1], -y[ind2]), 2, dim=1))

    return emd


def evaluate(preds, gts, train_frames, loss_type="CD"):
    print(f"Prediction sequence {len(preds)}, gts sequence {len(gts)}")
    if len(preds) != len(gts):
        print("[Error]: The prediction sequence is not align with the gt sequence.")
        return
    print(f"Prediction pcd particles cnt {preds[0].shape[0]}, gt pcd particles cnt {gts[0].shape[0]}")
    max_f = len(preds)
    losses = []
    fit_loss = 0.0
    predict_loss = 0.0
    for f in tqdm(range(max_f), desc=f"Evaluate {loss_type} Loss"):
        # pcd1 = discretize(preds[f], 0.02)
        # pcd2 = discretize(gts[f], 0.02)
        # cd = chamfer_distance(preds[f], gts[f])
        # print(f"frames: {f}, cd: {cd}")

        # cd align with https://zlicheng.com/spring_gaus/
        n_sample = 2048 if loss_type == "EMD" else 8192
        pcd0 = preds[f]
        pcd1 = gts[f]
        n_sample = min(n_sample, pcd0.shape[0], pcd1.shape[0])

        pcd0 = pcd0[random.sample(range(pcd0.shape[0]), n_sample), :]
        pcd1 = pcd1[random.sample(range(pcd1.shape[0]), n_sample), :]

        if loss_type == "CD":
            loss = (chamfer_distance(pcd0[None], pcd1[None])[0] * 1e3).item()
        elif loss_type == "EMD":
            loss = emd_func(pcd0, pcd1).item()
        else:
            print("[Error]: undefined error type.")
        if f < train_frames:
            fit_loss += loss
        else:
            predict_loss += loss
        losses.append(loss)

    fit_loss /= train_frames
    if max_f - train_frames > 0.0:
        predict_loss /= max_f - train_frames
    print(f"{loss_type} loss train: {fit_loss}, {loss_type} loss predict: {predict_loss}")
    return fit_loss, predict_loss, losses


def extract_views(cameras):
    views = cameras.copy()
    fids = torch.unique(torch.stack([view.fid for view in views]))
    views = {i: [view for view in views if view.fid == fids[i]] for i in range(len(fids))}
    views = {i: sorted(v, key=lambda x: x.colmap_id) for i, v in views.items()}
    return views


if __name__ == "__main__":
    cur_dir = os.path.abspath(os.curdir)
    start_time = time.time()

    parser = ArgumentParser(description="Prediction")
    parser.add_argument("--gt_path", type=str)
    parser.add_argument("--load_iter", type=int, default=0)
    parser.add_argument("--force_train", type=int, default=0)
    parser.add_argument(
        "--novel_gravity",
        type=float,
        nargs=3,
        default=None,
        help="set novel gravity, e.g. --novel_gravity -5.0 -2.5 0.0",
    )
    parser.add_argument(
        "--novel_velocity",
        type=float,
        nargs=3,
        default=None,
        help="set novel init velocity, e.g. --novel_velocity 1.0 0.5 0.0",
    )
    parser.add_argument("--novel_material", type=str, default=None, help="path of novel material")

    mdl = ModelParams(parser)  # , sentinel=True)
    ppl = PipelineParams(parser)
    opt = OptimizationParams(parser)
    arg, phy, unk = get_combined_args(parser)

    dat = mdl.extract(arg)
    opt = opt.extract(arg)
    ppl = ppl.extract(arg)

    cfg = OmegaConf.load("config/default.yaml")
    cfg.merge_with(unk)

    # set novel gravity
    if arg.novel_gravity is not None:
        cfg.sim.gravity = list(arg.novel_gravity)

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

    bg_color = [1, 1, 1] if dat.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    model_path = Path(dat.model_path)
    obj_name = model_path.parts[-1]

    # 0. Load GT
    gts = load_gt_pcds(arg.gt_path)
    total_frames = len(gts)
    phy.n_frames = phy.get("n_frames", total_frames)

    # 1. Load initial position
    position = load_pcd_file(dat.model_path)

    # 2. Load initial velocity and constitutive models
    model = Estimator(position, cfg, dat, opt, ppl, phy)
    resume_path = model.load_weights(dat.model_path, iteration=arg.load_iter, phase="constitution")
    print(f"constitution parameters loaded from {resume_path}")

    if arg.novel_velocity is not None:
        novel_velocity = list(arg.novel_velocity)
        for i in range(3):
            model.velocity[:, i].data += novel_velocity[i]

    # 3. Re-train appearance
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
    # scene.gaussians._features_dc = scene.gaussians._features_dc.mean(0, keepdim=True).repeat_interleave(len(scene.gaussians._features_dc), dim=0)
    # scene.gaussians._features_rest = scene.gaussians._features_dc.mean(0, keepdim=True).repeat_interleave(len(scene.gaussians._features_dc), dim=0)
    train_views = extract_views(scene.getTrainCameras(scale=dat.res_scale))
    test_views = extract_views(scene.getTestCameras(scale=dat.res_scale))
    train_frames = min(phy.get("n_frames", len(gts)), len(train_views))
    print("train frames:", train_frames, "total frames:", total_frames)

    # 4. Load other constitutive models
    if arg.novel_material is not None:
        novel_path = arg.novel_material
        novel_resume_path = model.load_weights(novel_path, iteration=arg.load_iter, phase="constitution")
        novel_model_name = os.path.basename(novel_path)
        # novel_model_name = os.path.basename(os.path.dirname(novel_path))
        print(f"constitution parameters loaded from {novel_resume_path}")
        novel_model_path = model_path / f"img_{novel_model_name}"
        novel_model_path.mkdir(exist_ok=True)
        print(f"saving to {novel_model_path}")

        # 5. Predicting trajectory
        novel_d_xyz_list = simulate(model, total_frames, diff=True, save_ply=True, path=dat.model_path)

        # 6. Future frame prediction
        with torch.no_grad():
            view = test_views[0][0]
            for frame_idx in range(train_frames):
                results = render(view, scene.gaussians, ppl, background, d_xyz=novel_d_xyz_list[frame_idx])
                pred = results["render"]
                gt = view.original_image.cuda()
                torchvision.utils.save_image(pred, novel_model_path / f"{view.uid}_{frame_idx:05d}.png")
                torchvision.utils.save_image(
                    results["alpha"], novel_model_path / f"{view.uid}_{frame_idx:05d}_mask.png"
                )

        render_abs_path = novel_model_path.resolve()
        cmd = ["ffmpeg", "-y", "-framerate", "15", "-pattern_type", "glob", "-i", "*_?????.png", "-pix_fmt", "yuv420p"]
        os.chdir(render_abs_path)
        subprocess.run(cmd + ["render.gif"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(cmd + ["render.mp4"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.chdir(cur_dir)
    else:
        preds = simulate(model, total_frames, diff=False, save_ply=True, path=dat.model_path)

        # 5. Future frame prediction
        with torch.no_grad():
            for frame_idx in range(total_frames):
                if frame_idx >= len(test_views):
                    break
                for view in test_views[frame_idx]:
                    results = render(view, scene.gaussians, ppl, background, d_xyz=d_xyz_list[frame_idx])
                    pred = results["render"]
                    gt = view.original_image.cuda()
                    torchvision.utils.save_image(pred, model_path / "img_velocity" / f"{view.uid}_{frame_idx:05d}.png")
                    torchvision.utils.save_image(
                        results["alpha"], model_path / "img_velocity" / f"{view.uid}_{frame_idx:05d}_mask.png"
                    )

        render_abs_path = (model_path / "img_velocity").resolve()
        cmd = ["ffmpeg", "-y", "-framerate", "30", "-pattern_type", "glob", "-i", "*_?????.png", "-pix_fmt", "yuv420p"]

        os.chdir(render_abs_path)
        subprocess.run(cmd + ["render.gif"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(cmd + ["render.mp4"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
