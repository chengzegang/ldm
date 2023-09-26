import copy
from functools import partial
import glob
import math
import os
from itertools import chain
from typing import Literal, OrderedDict, Type

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from loguru import logger
from torch import Tensor, nn
from torch.cuda.amp import GradScaler
from torch.distributed.optim import ZeroRedundancyOptimizer as ZeRO
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_, parametrizations, parametrize
from torch.optim import Adam, RMSprop
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.checkpoint import checkpoint
from .. import data, models
from . import diffusion


@torch.no_grad()
def show_triple(
    real_a: Tensor,
    noised_a: Tensor,
    denoised_a: Tensor,
) -> plt.Figure:
    n_samples = real_a.shape[0]
    ratio = real_a.shape[2] / real_a.shape[3]
    n_rows = 3
    fig, axes = plt.subplots(
        n_rows, n_samples, figsize=(3 * n_samples, 3 * ratio * n_rows), squeeze=False
    )
    for i in range(n_samples):
        axes[0, i].imshow(TF.to_pil_image(real_a[i].clamp(0, 1)))
        axes[1, i].imshow(TF.to_pil_image(noised_a[i].clamp(0, 1)))
        axes[2, i].imshow(TF.to_pil_image(denoised_a[i].clamp(0, 1)))
        axes[0, i].set_title("A")
        axes[1, i].set_title("Noised A")
        axes[2, i].set_title("Denoised A")
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        axes[2, i].set_xticks([])
        axes[2, i].set_yticks([])
    return fig


def qz_loss(z: Tensor) -> Tensor:
    qz = torch.where(z > 0, 1, -1).type_as(z).detach()
    loss = F.mse_loss(z, qz)
    return loss


def sample(z: Tensor) -> Tensor:
    qz = torch.where(z.detach() > 0, 1, -1).type_as(z)
    qz = qz - z.detach() + z  # reparameterization trick
    qz = z
    return qz


def consume_pattern(model_state: dict, pattern: str):
    state = OrderedDict()
    for key, value in model_state.items():
        state[key.replace(pattern, "")] = value
    return state


def setup_log_dir(config: dict, state: dict):
    if state["rank"] == 0:
        logger.info(f'setup log dir: {config["train"]["log_dir"]}')
        os.makedirs(config["train"]["log_dir"], exist_ok=True)


def apply_weight_norm(module: nn.Module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        if hasattr(module, "weight") and not parametrize.is_parametrized(
            module, "weight"
        ):
            module = parametrizations.weight_norm(module, "weight")
    return module


def remove_parametrizations(module: nn.Module):
    if hasattr(module, "weight") and parametrize.is_parametrized(module, "weight"):
        module = parametrize.remove_parametrizations(module, "weight")
    return module


class LDMTrainer(object):
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    model: diffusion.Diffusion
    scaler: GradScaler
    swa_scheduler: SWALR
    ema_model: AveragedModel
    dataset: Dataset | IterableDataset
    dataloader: DataLoader
    epoch: int
    step: int
    lr: float
    loss: float
    rank: int
    world_size: int
    device: torch.device
    dtype: torch.dtype

    def __init__(self, config: dict, vae_config: dict, vae_ckpt: str):
        self.config = config
        self.vae_config = vae_config
        self.vae_ckpt = vae_ckpt
        self.setup_env(config)
        self.setup_model(config, vae_config, vae_ckpt)
        self.setup_optimizer(config)
        self.setup_ema(config)
        self.setup_data(config)
        self.dtype = torch.float16

        self._kl_weights = torch.linspace(
            config["train"]["min_kl_weight"],
            config["train"]["max_kl_weight"],
            config["train"]["kl_annealing_steps"],
        )

    @property
    def lr(self):
        if self.step >= self.config["train"]["swa_start"]:
            return self.swa_scheduler.get_last_lr()[0]
        else:
            return self.scheduler.get_last_lr()[0]

    @property
    def _kl_weight(self):
        return self._kl_weights[self.step % len(self._kl_weights)]

    def setup_env(self, config: dict):
        if "ddp" in config and config["ddp"]:
            dist.init_process_group(backend="nccl")
        self.rank = int(os.getenv("LOCAL_RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))
        self.device = torch.device("cuda", self.rank)
        self.epoch = 0
        self.step = 0
        if self.rank == 0:
            logger.info(f'setup log dir: {config["train"]["log_dir"]}')
            os.makedirs(config["train"]["log_dir"], exist_ok=True)

        print(f"rank: {self.rank}, world_size: {self.world_size}")

    def setup_model(self, config: dict, vae_config: dict, vae_ckpt: str):
        logger.info(f"rank: {self.rank}: setup model")

        ckpt = self.load_checkpoint(config)
        vae = (
            models.VAE.from_meta(vae_config)
            .to(self.device)
            .to(memory_format=torch.channels_last)
        )
        clip = models.OpenClip()
        vae.load_state_dict(
            torch.load(vae_ckpt, map_location=self.device), strict=False
        )
        vae.to(self.device).to(memory_format=torch.channels_last)
        denoiser = (
            models.Transformer2d(**config["denoiser"])
            .to(self.device)
            .to(memory_format=torch.channels_last)
        )
        for layer in denoiser.layers:
            layer._org_forward_impl = layer.forward
            layer.forward = partial(checkpoint, layer.forward, use_reentrant=False)
        if ckpt is not None:
            denoiser.eval()
            denoiser.load_state_dict(ckpt["denoiser"])
        scheduler = diffusion.LinearScheduler(**config["diffusion"])
        sampler = diffusion.DDPMSampler(scheduler, **config["diffusion"])
        self.model = diffusion.Diffusion(vae, clip, denoiser, scheduler, sampler)

    def setup_optimizer(self, config: dict):
        logger.info(f"rank: {self.rank}: setup optimizer")

        self.optimizer = Adam(
            self.model.denoiser.parameters(),
            **config["optimizer"],
        )

        self.scheduler = ExponentialLR(
            self.optimizer, gamma=config["scheduler"]["lr_decay"]
        )
        self.scaler = GradScaler()

    @torch.no_grad()
    def load_checkpoint(self, config: dict) -> dict | None:
        logger.info(f"rank: {self.rank}: load checkpoint")
        latest_ckpt = None
        try:
            if (
                "checkpoint" in config["train"]
                and config["train"]["checkpoint"] is not None
            ):
                latest_ckpt = config["train"]["checkpoint"]
            else:
                ckpts = glob.glob(os.path.join(config["train"]["log_dir"], "*.pt"))
                if len(ckpts) == 0:
                    return
                latest_ckpt = max(ckpts, key=os.path.getctime)
            latest_ckpt = torch.load(latest_ckpt, map_location=self.device)
            self.step = latest_ckpt["step"]
            self.epoch = latest_ckpt["epoch"]
        except Exception as e:
            logger.error(f"rank: {self.rank}: {e}")
            return
        return latest_ckpt

    @torch.no_grad()
    def save_checkpoint(self):
        self.model.denoiser.eval()
        if self.rank == 0:
            torch.save(
                {
                    "denoiser": consume_pattern(
                        self.model.denoiser.state_dict(), "module."
                    ),
                    "step": self.step,
                    "epoch": self.epoch,
                },
                os.path.join(self.config["train"]["log_dir"], "diffusion.pt"),
            )

    def setup_ema(self, config: dict):
        logger.info(f"rank: {self.rank}: setup ema")
        self.swa_scheduler = SWALR(
            self.optimizer,
            anneal_strategy="cos",
            anneal_epochs=config["train"]["swa_anneal_steps"],
            swa_lr=config["optimizer"]["lr"],
        )

        def ema_avg(averaged_model_parameter, model_parameter, num_averaged):
            return 0.1 * averaged_model_parameter + 0.9 * model_parameter

        if config["train"]["swa_start"] > 0:
            self.ema_model = AveragedModel(self.model, avg_fn=ema_avg, use_buffers=True)

    def setup_data(self, config: dict):
        logger.info(f"rank: {self.rank}: setup data")
        if config["dataset"]["name"] == "image_folder":
            self.dataset = data.ImageFolder(**config["dataset"])
        elif config["dataset"]["name"] == "laion2B":
            self.dataset = data.Laion2B(**config["dataset"])

        shuffle = config["dataloader"]["shuffle"]
        self.sampler = None
        if isinstance(self.dataset, IterableDataset):
            shuffle = False

        elif "ddp" in config and config["ddp"]:
            self.sampler = DistributedSampler(self.dataset, shuffle=shuffle)
            shuffle = False

        dataloader_config = config["dataloader"].copy()
        dataloader_config.pop("shuffle")
        self.dataloader = DataLoader(
            self.dataset, **dataloader_config, shuffle=shuffle, sampler=self.sampler
        )

    def backward_step(self, loss: Tensor):
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if (
            self.config["train"]["swa_start"] > 0
            and self.step >= self.config["train"]["swa_start"]
        ):
            self.swa_scheduler.step()
            self.ema_model.update_parameters(self.model)
        else:
            self.scheduler.step()
        self.step += 1

    def diffusion_forward(self, x: Tensor, mask: Tensor, t: Tensor) -> dict:
        with torch.autocast("cuda", self.dtype):
            output = self.model(x, mask, t)
        self.backward_step(output["loss"])
        return output

    @torch.no_grad()
    def eval_step(self, image: Tensor, output: dict) -> dict:
        self.model.eval()
        self.save_checkpoint()
        fig = show_triple(image[:3], output["noised"][:3], output["output"][:3])
        fig.savefig(os.path.join(self.config["train"]["log_dir"], "result.png"))
        plt.close(fig)

    def train_step(self, image: Tensor):
        output = None
        self.model.train()
        output = self.diffusion_forward(
            image.to(self.device).contiguous(memory_format=torch.channels_last),
            torch.zeros_like(image)
            .to(self.device)
            .contiguous(memory_format=torch.channels_last),
            torch.randint(
                0, len(self.model.scheduler), (image.shape[0],), device=self.device
            ),
        )
        return output

    def verbose(self, output: dict):
        if self.rank == 0:
            desc = f"Epoch: {self.epoch}, Step: {self.step}, LR: {self.lr:.4e}"
            for key, value in output.items():
                key = key[0].upper() + key[1:]
                if isinstance(value, Tensor):
                    if torch.numel(value) == 1:
                        desc += f", {key}: {value.item():.4f}"
                else:
                    desc += f", {key}: {value:.4f}"
            logger.info(desc)

    def start(self):
        for epoch in range(self.epoch, self.config["train"]["total_epochs"]):
            self.epoch = epoch
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            for batch_idx, image in enumerate(self.dataloader):
                image = image.to(self.device).contiguous(
                    memory_format=torch.channels_last
                )

                output = self.train_step(image)
                self.verbose(output)
                if (
                    torch.isfinite(output["loss"])
                    and self.step % self.config["train"]["save_every"] == 0
                ):
                    self.eval_step(image, output)
