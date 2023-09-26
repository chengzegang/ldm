from abc import abstractmethod
from typing import List, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

from ..models import VAE, OpenClip


class DiffusionScheduler(object):
    betas: Tensor
    alphas: Tensor
    alphas_cumprod: Tensor
    alphas_cumprod_prev: Tensor
    timesteps: Tensor

    def __len__(self):
        return len(self.timesteps)


class LinearScheduler(DiffusionScheduler):
    def __init__(
        self,
        total_steps: int = 1000,
        min_beta: float = 0.02,
        max_beta: float = 0.000001,
        **kwargs
    ):
        self.betas = torch.linspace(min_beta, max_beta, total_steps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod, (1, 0), value=1.0)
        self.timesteps = torch.arange(total_steps)


class DiffusionSampler(object):
    def __init__(self, scheduler: DiffusionScheduler, **kwargs):
        self.scheduler = scheduler

    def add_noise(self, x: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        sample_coeffs = torch.sqrt(self.scheduler.alphas_cumprod_prev.to(t.device)[t])
        sample_coeffs = sample_coeffs[:, None, None, None]
        noise_coeffs = torch.sqrt(
            1 - self.scheduler.alphas_cumprod_prev.to(t.device)[t]
        )
        noise_coeffs = noise_coeffs[:, None, None, None]
        x = x * sample_coeffs + noise * noise_coeffs
        return x

    def forward(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        noise = torch.randn_like(x)
        x = self.add_noise(x, noise, t)
        return x, noise

    @abstractmethod
    def _backward_scaled(
        self, x: Tensor, epsilon: Tensor, t: Tensor, t_prev: Tensor
    ) -> Tuple[Tensor, Tensor]:
        ...

    def backward(
        self, x: Tensor, epsilon: Tensor, t: Tensor, t_prev: Tensor
    ) -> Tuple[Tensor, Tensor]:
        x_prev, x0 = self._backward_scaled(x, epsilon, t, t_prev)
        x_prev = torch.where(t[:, None, None, None].expand_as(x) > 0, x, x_prev)
        return x_prev, x0


class DDPMSampler(DiffusionSampler):
    def _backward_scaled(
        self, x: Tensor, epsilon: Tensor, t: Tensor, t_prev: Tensor
    ) -> Tuple[Tensor, Tensor]:
        noise_curr_coeff = torch.sqrt(1 - self.scheduler.alphas_cumprod.to(t.device)[t])
        noise_prev_coeff = torch.sqrt(
            1 - self.scheduler.alphas_cumprod.to(t.device)[t_prev]
        )
        sample_curr_coeff = torch.sqrt(self.scheduler.alphas_cumprod.to(t.device)[t])
        sample_prev_coeff = torch.sqrt(
            self.scheduler.alphas_cumprod.to(t.device)[t_prev]
        )

        noise_curr_coeff = noise_curr_coeff[:, None, None, None]
        noise_prev_coeff = noise_prev_coeff[:, None, None, None]
        sample_curr_coeff = sample_curr_coeff[:, None, None, None]
        sample_prev_coeff = sample_prev_coeff[:, None, None, None]

        x0 = (x - epsilon * noise_curr_coeff) / sample_curr_coeff
        x_prev = (x - epsilon * noise_prev_coeff) / sample_prev_coeff
        return x_prev, x0


class Diffusion(nn.Module):
    def __init__(
        self,
        vae: VAE,
        clip: OpenClip,
        denoiser: nn.Module,
        scheduler: DiffusionScheduler,
        sampler: DiffusionSampler,
    ):
        super().__init__()
        self.vae = vae
        self.clip = clip
        self.denoiser = denoiser
        self.scheduler = scheduler
        self.sampler = sampler

    def forward(
        self, x: Tensor, mask: Tensor, t: Tensor, text: List[str] | None = None
    ) -> dict:
        self.vae.eval()
        self.clip.eval()
        self.denoiser.train()
        clip_emb = self.clip.encode_image(x)
        if text is not None:
            text_emb = self.clip.encode_text(text)
            clip_emb = torch.cat([clip_emb, text_emb], dim=1)
        else:
            clip_emb = clip_emb.repeat(1, 2, 1)
        mask_emb = self.vae.encode(mask).sample()
        with torch.no_grad():
            z = self.vae.encode(x).sample()
            noised_z, noise = self.sampler.forward(z, t)
        pred_noise = self.denoiser(noised_z, mask_emb, clip_emb, t)
        with torch.no_grad():
            pred_z = self.sampler.backward(noised_z, pred_noise, t, t - 1)[1]
            xh = self.vae.decode(pred_z)
        with torch.no_grad():
            noised_xh = self.vae.decode(noised_z)
        loss = F.mse_loss(pred_noise, noise)
        return dict(output=xh, noised=noised_xh, loss=loss)
