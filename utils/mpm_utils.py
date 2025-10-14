import torch

from nclaw.warp.svd import svd


def compute_R_from_F(F):
    U, S, Vh = svd(F)
    Up = U.clone()
    Up[torch.linalg.det(U) < 0, :, 2] *= -1
    Vhp = Vh.clone()
    Vhp[torch.linalg.det(Vh) < 0, 2] *= -1
    R = (Up @ Vhp).transpose(-2, -1)
    return R


def compute_cov_from_F(F, init_cov):
    init_cov33 = torch.zeros_like(F)
    cov = torch.zeros_like(init_cov)

    init_cov33[:, 0, 0] = init_cov[:, 0]
    init_cov33[:, 0, 1] = init_cov[:, 1]
    init_cov33[:, 0, 2] = init_cov[:, 2]
    init_cov33[:, 1, 0] = init_cov[:, 1]
    init_cov33[:, 1, 1] = init_cov[:, 3]
    init_cov33[:, 1, 2] = init_cov[:, 4]
    init_cov33[:, 2, 0] = init_cov[:, 2]
    init_cov33[:, 2, 1] = init_cov[:, 4]
    init_cov33[:, 2, 2] = init_cov[:, 5]

    cov33 = F @ init_cov33 @ F.transpose(-2, -1)

    cov[:, 0] = cov33[:, 0, 0]
    cov[:, 1] = cov33[:, 0, 1]
    cov[:, 2] = cov33[:, 0, 2]
    cov[:, 3] = cov33[:, 1, 1]
    cov[:, 4] = cov33[:, 1, 2]
    cov[:, 5] = cov33[:, 2, 2]

    return cov
