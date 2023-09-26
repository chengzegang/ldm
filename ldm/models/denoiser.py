import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List
from ..modules.attention import QuantCrossAttentionLayer
from ..modules.unet import UNetEncoder, UNetDecoder
from ..modules.convolutions import QuantConv2d


class Transformer2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        latent_size: int,
        clip_emb_size: int,
        attn_head_size: int = 128,
        num_transformer_layers: int = 6,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_size = latent_size
        self.clip_emb_size = clip_emb_size
        self.layers = nn.ModuleList()
        self.latent_emb = QuantConv2d(in_channels * 2, in_channels, kernel_size=1)
        self.clip_emb = nn.Linear(clip_emb_size, latent_size)
        self.time_emb = nn.Embedding(10000, latent_size)
        self.encoder = UNetEncoder(in_channels, channels, latent_size)
        self.decoder = UNetDecoder(in_channels, channels, latent_size)
        for i in range(num_transformer_layers):
            self.layers.append(QuantCrossAttentionLayer(latent_size, attn_head_size))

    def forward(
        self, latent: Tensor, mask: Tensor, cond_emb: Tensor, t_emb: Tensor
    ) -> Tensor:
        latent = torch.cat([latent, mask], dim=1)
        latent_emb = self.latent_emb(latent)

        cond_emb = self.clip_emb(cond_emb)
        t_emb = self.time_emb(t_emb).unsqueeze(1)
        cond_emb = torch.cat([cond_emb, t_emb], dim=1)

        latent_emb = self.encoder(latent_emb)
        shape = latent_emb.shape
        latent_emb = latent_emb.flatten(2).transpose(-1, -2)

        for layer in self.layers:
            latent_emb = layer(latent_emb, cond_emb)

        latent_emb = latent_emb.transpose(-1, -2).reshape(shape)
        latent = self.decoder(latent_emb)

        return latent
