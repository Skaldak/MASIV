import os
from argparse import Namespace

import numpy as np
import torch
import warp as wp
from kornia import tensor_to_image
from omegaconf import DictConfig
from torch import nn
from torch.utils.checkpoint import checkpoint
from tqdm import trange

from gaussian_renderer import render
from nclaw.material import InvariantFullMetaElasticity, InvariantFullMetaPlasticity
from nclaw.sim import MPMModelBuilder, MPMStaticsInitializer, MPMInitData, MPMDiffSim
from simulator.constitution import MetaOptimizer
from utils.loss_utils import l1_loss, ssim
from utils.system_utils import searchForMaxIteration


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class Estimator(nn.Module):
    def __init__(
        self, position, cfg: DictConfig, dat: Namespace, opt: Namespace, ppl: Namespace, phy: Namespace, device="cuda"
    ):
        super(Estimator, self).__init__()

        self.cfg = cfg
        self.dat = dat
        self.opt = opt
        self.ppl = ppl
        self.phy = phy
        self.device = device

        cfg.sim.max_frames = None
        cfg.sim.frame_dt = 1.0 / phy.fps
        cfg.sim.dt = cfg.sim.frame_dt / cfg.sim.substeps
        cfg.sim.num_steps = phy.n_frames * cfg.sim.substeps
        cfg.sim.gravity = (np.asarray(phy.gravity) * np.asarray(phy.get("scale", 1.0))).tolist()

        cfg.env.size = cfg.sim.size
        cfg.env.center = cfg.sim.center
        cfg.env.num_steps = cfg.env.max_steps = cfg.sim.substeps

        model = MPMModelBuilder().parse_cfg(cfg.sim).finalize(wp.get_device(device), requires_grad=True)
        statics_initializer = MPMStaticsInitializer(model)
        init_data = MPMInitData.get(cfg.env, Struct(get_xyz=position))
        statics_initializer.add_group(init_data)
        statics = statics_initializer.finalize()

        self.scale = torch.tensor([phy.get("scale", 1.0)], device=device)
        self.offset = torch.tensor([phy.get("offset", [0.0, 0.0, 0.0])], device=device)
        self.offset -= (torch.tensor([phy.bc.ground[0]]) * torch.tensor([phy.bc.ground[1]])).to(device)
        self.offset += torch.tensor([cfg.sim.size / 2 - cfg.sim.center]).to(device)

        self.sim = MPMDiffSim(model, statics)
        self.position = position.to(device)
        if self.cfg.env.simplify_velocity:
            self.velocity = nn.Parameter(torch.zeros_like(position, device=device)[:1])
        else:
            self.velocity = nn.Parameter(torch.zeros_like(position, device=device))
        self.elasticity = InvariantFullMetaElasticity(cfg.meta.elasticity).to(device)
        self.plasticity = InvariantFullMetaPlasticity(cfg.meta.plasticity).to(device)

        if self.cfg.env.pretrain == "jelly":
            self.load_pretrained_weights("pretrained/jelly_0300.pt")
        elif self.cfg.env.pretrain == "plasticine":
            self.load_pretrained_weights("pretrained/plasticine_0300.pt")
        elif self.cfg.env.pretrain == "sand":
            self.load_pretrained_weights("pretrained/sand_0300.pt")

        self.optimizer = MetaOptimizer(cfg, self)

    def initialize_parameters(self):
        self.sim.model.constant.dt = self.cfg.sim.dt
        self.cfg.env.num_steps = self.cfg.env.max_steps
        x = self.position
        v = self.velocity
        C = torch.zeros(x.shape[0], 3, 3).to(x)
        F = torch.eye(3)[None].repeat_interleave(x.shape[0], dim=0).to(x)
        return x, v, C, F

    def forward_loss(self, data, mode="train"):
        dt = self.cfg.sim.dt
        num_steps = self.cfg.env.max_steps
        x, v, C, F = None, None, None, None
        losses, losses_geo, losses_pho, losses_traj, losses_velo = [], [], [], [], []

        while True:
            for idx in trange(self.cfg.sim.max_frames):
                if idx == 0:
                    # no loss for the first frame
                    x, v, C, F = self.initialize_parameters()
                    if len(v) != len(x):
                        v = v.repeat_interleave(x.shape[0], dim=0)
                    self.sim.model.constant.dt = dt
                    self.cfg.env.num_steps = num_steps
                    continue
                else:
                    x, v, C, F, xs, vs = checkpoint(self.advance, x, v, C, F, use_reentrant=True)
                    if self.cfg.env.satisfy_cfl and not self.cfl_satisfy(self.normalize(v, offset=False)):
                        losses = []
                        dt /= 2
                        num_steps *= 2
                        print(f"cfl condition dissatisfy, shrink dt {dt}, step cnt {num_steps}")
                        break
                loss = torch.zeros(1).to(x)
                xs_gt = []
                with torch.no_grad():
                    t0 = torch.as_tensor(data[f"{mode}_views"][idx - 1][0].fid).to(x)[None]
                    t1 = torch.as_tensor(data[f"{mode}_views"][idx][0].fid).to(x)[None]
                    for step in range(1, num_steps + 1):
                        t = t0 + (t1 - t0) / num_steps * step
                        dx, *_ = data["deform"].step(data["canonical"], t)
                        gt = data["canonical"] + dx
                        xs_gt.append(gt)
                    xs_gt = torch.stack(xs_gt)
                    vs_gt = (xs_gt[1:] - xs_gt[:-1]) / dt
                if self.cfg.meta.w_traj > 0:
                    loss_traj = self.cfg.meta.w_traj * l1_loss(xs, xs_gt)
                    losses_traj.append(loss_traj)
                else:
                    loss_traj = 0.0
                if self.cfg.meta.w_velo > 0:
                    loss_velo = self.cfg.meta.w_velo * l1_loss((vs[1:] + vs[:-1]) / 2, vs_gt)
                    losses_velo.append(loss_velo)
                else:
                    loss_velo = 0.0
                loss_geo = self.phy.w_geo * (loss_traj + loss_velo)
                losses_geo.append(loss_geo)
                loss += losses_geo[-1]
                if self.optimizer.phase == "constitution":
                    if self.phy.w_img > 0 or self.phy.w_alp > 0:
                        loss_img, loss_alp = 0, 0
                        for view in data[f"{mode}_views"][idx]:
                            results = render(
                                view,
                                data["scene"].gaussians,
                                self.ppl,
                                data["background"],
                                d_xyz=x - data["scene"].gaussians.get_xyz,
                            )
                            image_pred, alpha_pred = results["render"], results["alpha"]
                            image_gt, alpha_gt = view.original_image.to(image_pred), view.gt_alpha_mask.to(alpha_pred)
                            mask = ((alpha_pred > 0) + (alpha_gt > 0)).nonzero()
                            y_max, y_min = mask[:, 1].max(), mask[:, 1].min()
                            x_max, x_min = mask[:, 2].max(), mask[:, 2].min()
                            if self.phy.w_img > 0:
                                masked_image_pred = image_pred[:, y_min:y_max, x_min:x_max]
                                masked_image_gt = image_gt[:, y_min:y_max, x_min:x_max]
                                l_1 = l1_loss(masked_image_pred, masked_image_gt)
                                l_ssim = 1.0 - ssim(masked_image_pred, masked_image_gt)
                                loss_img += (1.0 - self.opt.lambda_dssim) * l_1 + self.opt.lambda_dssim * l_ssim
                            if self.phy.w_alp > 0:
                                masked_alpha_pred = alpha_pred[:, y_min:y_max, x_min:x_max]
                                masked_alpha_gt = alpha_gt[:, y_min:y_max, x_min:x_max]
                                loss_alp += l1_loss(masked_alpha_pred, masked_alpha_gt)
                        loss_pho = self.phy.w_img * loss_img + self.phy.w_alp * loss_alp
                        loss_pho = loss_pho / len(data[f"{mode}_views"][idx])
                        losses_pho.append(loss_pho)
                        loss += losses_pho[-1]
                losses.append(loss)
            if len(losses) == self.cfg.sim.max_frames - 1:
                break

        loss = torch.stack(losses).mean()
        loss_geo = torch.stack(losses_geo).mean().detach()
        loss_pho = torch.stack(losses_pho).mean().detach() if len(losses_pho) > 0 else 0
        loss_traj = torch.stack(losses_traj).mean().detach() if len(losses_traj) > 0 else 0
        loss_velo = torch.stack(losses_velo).mean().detach() if len(losses_velo) > 0 else 0

        return loss, loss_geo, loss_pho, loss_traj, loss_velo

    @torch.no_grad()
    def forward_render(self, data, mode="test"):
        x, v, C, F = None, None, None, None
        frames = {i.colmap_id: [] for i in data[f"{mode}_views"][0]}

        for frame_idx in trange(len(data[f"{mode}_views"])):
            if frame_idx == 0:
                x, v, C, F = self.initialize_parameters()
                if len(v) != len(x):
                    v = v.repeat_interleave(x.shape[0], dim=0)
            else:
                x, v, C, F, *_ = self.advance(x, v, C, F)
            for view in data[f"{mode}_views"][frame_idx]:
                d_xyz = x - data["scene"].gaussians.get_xyz
                results = render(view, data["scene"].gaussians, self.ppl, data["background"], d_xyz=d_xyz)
                image_pred, alpha_pred = tensor_to_image(results["render"]), tensor_to_image(results["alpha"])
                alpha_pred = alpha_pred[..., None].repeat(3, axis=2)
                image_gt, alpha_gt = tensor_to_image(view.original_image), tensor_to_image(view.gt_alpha_mask)
                alpha_gt = alpha_gt[..., None].repeat(3, axis=2)
                d_xyz, *_ = data["deform"].step(data["canonical"], view.fid.to(x)[None])
                d_xyz = data["canonical"] + d_xyz - data["scene"].gaussians.get_xyz
                results = render(view, data["scene"].gaussians, self.ppl, data["background"], d_xyz=d_xyz)
                image_rec, alpha_rec = tensor_to_image(results["render"]), tensor_to_image(results["alpha"])
                alpha_rec = alpha_rec[..., None].repeat(3, axis=2)
                frame = np.concatenate([image_pred, image_rec, image_gt, alpha_pred, alpha_rec, alpha_gt], axis=1)
                frame = (frame * 255).astype(np.uint8)
                frames[view.colmap_id].append(frame)

        return frames

    def advance(self, x, v, C, F):
        xs, vs = [], []
        x_curr, v_curr = self.normalize(x), self.normalize(v, offset=False)
        xw_curr, vw_curr, C_curr, F_curr = x, v, C, F
        for idx in range(self.cfg.env.num_steps):
            if not self.cfl_satisfy(vw_curr):
                v_max = self.sim.model.constant.dx / self.sim.model.constant.dt
                v_curr = v_curr.nan_to_num(nan=0.0, posinf=v_max, neginf=-v_max).clip(-v_max, v_max)
            stress = self.elasticity(F_curr)
            x_next, v_next, C_next, F_trial = self.sim(x_curr, v_curr, C_curr, F_curr, stress)
            F_next = self.plasticity(F_trial)
            x_curr, v_curr, C_curr, F_curr = x_next, v_next, C_next, F_next
            xw_curr, vw_curr = self.denormalize(x_next), self.denormalize(v_next, offset=False)
            xs.append(xw_curr)
            vs.append(vw_curr)
        xs, vs = torch.stack(xs), torch.stack(vs)
        return xw_curr, vw_curr, C_curr, F_curr, xs, vs

    def cfl_satisfy(self, v):
        if v.isnan().any():
            print("[WARNING] v is nan")
            return False
        if v.abs().max() * self.sim.model.constant.dt > self.sim.model.constant.dx:
            print("[WARNING] v is too high")
            return False
        return True

    def normalize(self, x, scale=True, offset=True):
        s = self.scale if scale else 1
        o = self.offset if offset else 0
        if isinstance(x, wp.vec3):
            x = torch.as_tensor(x, device=self.device) * s + o
            return wp.vec3(*x.cpu().numpy().tolist())
        else:
            return x * s + o

    def denormalize(self, p, scale=True, offset=True):
        s = self.scale if scale else 1
        o = self.offset if offset else 0
        if isinstance(p, wp.vec3):
            p = (torch.as_tensor(p, device=self.device) - o) / s
            return wp.vec3(*p.cpu().numpy().tolist())
        else:
            return (p - o) / s

    def set_phase(self, phase, max_frames):
        self.optimizer.phase = phase
        self.cfg.sim.max_frames = max_frames

    def save_weights(self, model_path, iteration=-1):
        out_weights_path = os.path.join(model_path, f"{self.optimizer.phase}/iteration_{iteration}")
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(out_weights_path, "dynamics.pth"))

    def load_pretrained_weights(self, model_path):
        weights = torch.load(model_path, weights_only=True, map_location="cpu")
        self.elasticity.load_state_dict(weights["elasticity"], strict=False)
        self.plasticity.load_state_dict(weights["plasticity"], strict=False)

    def load_weights(self, model_path, iteration=0, phase=None):
        if iteration == 0:
            load_iter = searchForMaxIteration(os.path.join(model_path, phase))
        else:
            load_iter = iteration
        load_path = os.path.join(model_path, f"{phase}/iteration_{load_iter}/dynamics.pth")
        weights = torch.load(load_path, weights_only=True, map_location="cpu")
        if phase == "velocity":
            self.velocity.data = weights["velocity"].to(self.device)
        elif phase == "constitution":
            velocity = self.velocity.data.clone()
            self.load_state_dict(weights, strict=False)
            self.velocity.data = velocity
        else:
            self.load_state_dict(weights, strict=False)
        return load_path
