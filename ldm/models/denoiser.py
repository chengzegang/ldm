import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List
from ..modules.attention import QuantCrossAttentionLayer
from ..modules.unet import UNetEncoder, UNetDecoder, UNet
from ..modules.convolutions import QuantConv2d


class AttentionUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: List[int],
        clip_emb_size: int,
        attn_head_size: int = 128,
        num_transformer_layers: int = 6,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_size = clip_emb_size
        self.clip_emb_size = clip_emb_size
        self.layers = nn.ModuleList()
        self.latent_emb = QuantConv2d(in_channels * 2, in_channels, kernel_size=1)
        self.clip_emb = nn.Linear(clip_emb_size, clip_emb_size)
        self.time_emb = nn.Embedding(10000, clip_emb_size)
        self.unet = UNet(
            in_channels,
            out_channels,
            channels,
            clip_emb_size,
            clip_emb_size,
            num_transformer_layers=num_transformer_layers,
        )
        self.time_mlp = nn.Sequential()
        for _ in range(3):
            self.time_mlp.append(nn.Linear(clip_emb_size, clip_emb_size))
            self.time_mlp.append(nn.LayerNorm(clip_emb_size))
            self.time_mlp.append(nn.SiLU(True))

    def forward(
        self,
        latent: Tensor,
        t: Tensor,
        mask: Tensor,
        cond: Tensor,
    ) -> Tensor:
        latent = torch.cat([latent, mask], dim=1)
        latent_emb = self.latent_emb(latent)
        cond_emb = self.clip_emb(cond)
        t_emb = self.time_emb(t.long()).unsqueeze(1)
        t_emb = self.time_mlp(t_emb)
        cond_emb = torch.cat([cond_emb, t_emb], dim=1)
        latent = self.unet(latent_emb, cond_emb)

        return latent
