from copy import deepcopy

import torch
import torch.nn as nn

from utils.transform_utils import rot6d_to_rotmat, euler_to_quat, quat_to_rot6d


class Register(nn.Module):
    def __init__(self, cfg, init_data):
        super().__init__()
        self.name = type(self).__name__
        self.cfg = cfg

        euler = torch.tensor(init_data.rotation, dtype=torch.float32) * torch.pi / 180
        quat = euler_to_quat(euler)
        rot6d = quat_to_rot6d(quat)

        self.r = nn.Parameter(rot6d, requires_grad=True)
        self.t = nn.Parameter(torch.tensor(init_data.translation, dtype=torch.float32), requires_grad=True)
        self.s = nn.Parameter(torch.tensor(init_data.scale, dtype=torch.float32), requires_grad=True)

    def training_setup(self, training_args):
        l = [
            {"params": [self.r], "lr": training_args.rotation_lr, "name": "r"},
            {"params": [self.t], "lr": training_args.translation_lr, "name": "t"},
            {"params": [self.s], "lr": training_args.scale_lr, "name": "s"},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    @property
    def get_scale(self):
        return deepcopy(self.s).detach()

    def forward(self, xyz):
        R = rot6d_to_rotmat(self.r)

        origin = torch.mean(xyz, dim=0, keepdim=True)
        xyz = self.s * (xyz - origin)  # + origin

        xyz = (R @ xyz.transpose(0, 1)).transpose(0, 1) + self.t.unsqueeze(0)

        return xyz
