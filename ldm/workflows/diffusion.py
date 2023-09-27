from typing import List
import torch
from torch import nn
from torch import Tensor
from ..models import VAE, OpenClip
from .openai_diffusion import GaussianDiffusion, LossType, ModelMeanType, ModelVarType
import numpy as np


class Diffusion(nn.Module):
    def __init__(
        self,
        vae: VAE,
        clip: OpenClip,
        model: nn.Module,
        train_steps: int = 1000,
        beta_min: float = 0.0001,
        beta_max: float = 0.02,
        device: str = "cuda",
    ):
        super().__init__()
        self.vae = vae
        self.clip = clip
        self.train_steps = train_steps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.model = model
        self.device = device
        self.diffusion = GaussianDiffusion(
            betas=np.linspace(beta_max, beta_min, train_steps),
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.LEARNED_RANGE,
            loss_type=LossType.RESCALED_KL,
        )

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        t: Tensor,
        text: List[str] | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> dict:
        self.vae.eval()
        self.clip.eval()
        self.model.train()
        with torch.no_grad(), torch.autocast("cuda", dtype):
            clip_emb = self.clip.encode_image(x)
            x = self.vae.encode(x).sample()
            if text is not None:
                text_emb = self.clip.encode_text(text)
                clip_emb = torch.cat([clip_emb, text_emb], dim=1)
            else:
                clip_emb = clip_emb.repeat(1, 2, 1)
            mask_emb = self.vae.encode(mask).sample()
        with torch.autocast("cuda", torch.float32):
            output = self.diffusion.training_losses(
                self.model, x, t, model_kwargs=dict(cond=clip_emb, mask=mask_emb)
            )
        return output["loss"].mean()

    @torch.no_grad()
    def eval_forward(
        self,
        x: Tensor,
        mask: Tensor,
        t: Tensor,
        text: List[str] | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> dict:
        self.vae.eval()
        self.clip.eval()
        self.model.eval()
        input = x
        with torch.no_grad(), torch.autocast("cuda", dtype):
            clip_emb = self.clip.encode_image(x)
            x = self.vae.encode(x).sample()
            if text is not None:
                text_emb = self.clip.encode_text(text)
                clip_emb = torch.cat([clip_emb, text_emb], dim=1)
            else:
                clip_emb = clip_emb.repeat(1, 2, 1)
            mask_emb = self.vae.encode(mask).sample()
            noised = self.diffusion.q_sample(x, t)

        denoised = self.diffusion.ddim_sample_loop(
            self.model,
            x.shape,
            model_kwargs=dict(cond=clip_emb, mask=mask_emb),
            device=self.device,
            progress=True,
        )
        denoised = self.vae.decode(denoised)
        noised = self.vae.decode(noised)
        return dict(input=input, noised=noised, denoised=denoised)
