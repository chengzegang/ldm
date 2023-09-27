from abc import abstractmethod
from functools import partial
from typing import List, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
import k_diffusion as kdf
from ..models import VAE, OpenClip


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class DiffusionScheduler(nn.Module):
    timesteps: int
    beta_min: float
    beta_max: float
    betas: Tensor
    alphas: Tensor
    alpha_cumprod: Tensor
    alpha_cumprod_prev: Tensor


class DiffusionSampler(nn.Module):
    diffusion_scheduler: DiffusionScheduler

    @abstractmethod
    def diffusion_forward(self, sample: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        ...

    @abstractmethod
    def diffusion_backward(
        self, sample: Tensor, noise: Tensor, t: Tensor, t_prev: Tensor
    ):
        ...
