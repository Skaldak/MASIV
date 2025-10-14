import torch
import warp as wp
from omegaconf import DictConfig
from torch import nn, Tensor


class Constitution(nn.Module):
    def __init__(self, cfg: DictConfig, device="cuda"):
        super(Constitution, self).__init__()

        self.cfg = cfg
        self.device = device


class MetaOptimizer:
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.phase = None

        self.velocity_optimizer = torch.optim.RAdam(
            [{"name": "velocity", "params": model.velocity}],
            lr=cfg.velocity.lr,
        )
        self.elasticity_optimizer = torch.optim.RAdam(
            [{"name": "elasticity", "params": model.elasticity.parameters()}],
            lr=cfg.meta.elasticity_lr,
            weight_decay=cfg.meta.elasticity_wd,
        )
        self.plasticity_optimizer = torch.optim.RAdam(
            [{"name": "plasticity", "params": model.plasticity.parameters()}],
            lr=cfg.meta.plasticity_lr,
            weight_decay=cfg.meta.plasticity_wd,
        )

        self.velocity_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.velocity_optimizer,
            cfg.velocity.scheduler.max_steps,
            eta_min=cfg.velocity.scheduler.learning_rate_alpha * cfg.velocity.lr,
        )
        self.elasticity_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.elasticity_optimizer,
            cfg.meta.elasticity_scheduler.max_steps,
            cfg.meta.elasticity_scheduler.learning_rate_alpha * cfg.meta.elasticity_lr,
        )
        self.plasticity_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.plasticity_optimizer,
            cfg.meta.plasticity_scheduler.max_steps,
            cfg.meta.plasticity_scheduler.learning_rate_alpha * cfg.meta.plasticity_lr,
        )

    @property
    def param_groups(self):
        if self.phase == "velocity":
            return self.velocity_optimizer.param_groups
        elif self.phase == "constitution":
            return self.elasticity_optimizer.param_groups + self.plasticity_optimizer.param_groups

    def step(self):
        messages = []
        if self.phase == "velocity":
            self.velocity_optimizer.step()
        elif self.phase == "constitution":
            elasticity_params = self.elasticity_optimizer.param_groups[0]["params"]
            plasticity_params = self.plasticity_optimizer.param_groups[0]["params"]
            elasticity_grad_norm = nn.utils.clip_grad_norm_(
                elasticity_params, max_norm=self.cfg.meta.elasticity_grad_max_norm, error_if_nonfinite=True
            )
            plasticity_grad_norm = nn.utils.clip_grad_norm_(
                plasticity_params, max_norm=self.cfg.meta.plasticity_grad_max_norm, error_if_nonfinite=True
            )
            messages += [f"e-gn: {elasticity_grad_norm.item():.4e}", f"p-gn: {plasticity_grad_norm.item():.4e}"]
            self.elasticity_optimizer.step()
            self.plasticity_optimizer.step()
        return messages

    def update_learning_rate(self, epoch=None):
        if self.phase == "velocity":
            self.velocity_scheduler.step(epoch=epoch)
        elif self.phase == "constitution":
            self.elasticity_scheduler.step(epoch=epoch)
            self.plasticity_scheduler.step(epoch=epoch)

    def zero_grad(self, set_to_none=False):
        self.velocity_optimizer.zero_grad(set_to_none=set_to_none)
        self.elasticity_optimizer.zero_grad(set_to_none=set_to_none)
        self.plasticity_optimizer.zero_grad(set_to_none=set_to_none)
