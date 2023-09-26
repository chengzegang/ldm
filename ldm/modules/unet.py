from typing import List, Optional

from torch import Tensor, nn
from .attention import AttentionLayer2d, QuantAttentionLayer2d
from .convolutions import ResidualBlock, QuantConvTranspose2d, QuantConv2d
import torch


class ConvDown(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, eps)
        self.down = QuantConv2d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.res(x)
        x = self.down(x)
        return x


class ConvUp(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.up = QuantConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.res = ResidualBlock(out_channels, out_channels, eps)

    def forward(self, x: Tensor) -> Tensor:
        x = self.up(x)
        x = self.res(x)
        return x


class UNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        latent_size: int,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.latent_size = latent_size
        self.layers = nn.ModuleList()
        self.in_conv = QuantConv2d(in_channels, channels[0], kernel_size=1)
        for i in range(len(channels) - 1):
            self.layers.append(ConvDown(channels[i], channels[i + 1], eps))
        self.layers.append(ResidualBlock(channels[-1], channels[-1], eps))
        self.attn = QuantAttentionLayer2d(
            channels[-1],
            128,
        )
        self.out_conv = QuantConv2d(channels[-1], latent_size, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_conv(x)
        for layer in self.layers:
            x = layer(x)
        x = self.attn(x)
        z = self.out_conv(x)
        return z


class UNetDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        channels: List[int],
        latent_size: int,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.out_channels = out_channels
        channels = channels[::-1]
        self.in_conv = QuantConv2d(latent_size, channels[0], kernel_size=1)
        self.attn = QuantAttentionLayer2d(
            channels[0],
            128,
        )
        self.layers = nn.ModuleList()
        self.layers.append(ResidualBlock(channels[0], channels[0], eps))
        for i in range(len(channels) - 1):
            self.layers.append(ConvUp(channels[i], channels[i + 1], eps))
        self.out_norm = nn.InstanceNorm2d(channels[-1], eps=eps)
        self.out = QuantConv2d(channels[-1], out_channels, kernel_size=1)

    def forward(self, qz: Tensor) -> Tensor:
        x = self.in_conv(qz)
        x = self.attn(x)
        for layer in self.layers:
            x = layer(x)
        x = self.out(x)
        return x
