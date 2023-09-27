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
        self.latent_emb = nn.Conv2d(in_channels * 2, clip_emb_size, kernel_size=1)
        self.cond_emb = nn.Linear(clip_emb_size, clip_emb_size)
        self.time_emb = nn.Linear(1, clip_emb_size)
        self.t2l_emb = nn.Conv2d(clip_emb_size * 2, clip_emb_size, 3, padding=1)
        self.unet = UNet(
            clip_emb_size,
            clip_emb_size,
            channels,
            clip_emb_size,
            clip_emb_size,
            num_transformer_layers=num_transformer_layers,
        )
        self.conv_out = nn.Conv2d(clip_emb_size, out_channels, 1)

    def forward(
        self,
        latent: Tensor,
        t: Tensor,
        cond: Tensor,
        mask: Tensor,
    ) -> Tensor:
        latent = torch.cat([latent, mask], dim=1)
        cond_emb = self.cond_emb(cond)
        t_emb = self.time_emb(torch.log(1 + t.unsqueeze(-1).type_as(latent)))

        latent_emb = self.latent_emb(latent)
        latent_emb = self.t2l_emb(
            torch.cat(
                [latent_emb, t_emb[:, :, None, None].expand_as(latent_emb)], dim=1
            )
        )

        cond_emb = torch.cat([cond_emb, t_emb.unsqueeze(1)], dim=1)

        latent = self.unet(latent_emb, cond_emb)
        latent = self.conv_out(latent)
        return latent
