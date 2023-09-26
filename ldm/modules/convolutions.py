from typing import Tuple
from torch import Tensor, nn

import torch
from torch.ao.quantization.quantize_fx import prepare_fx
from .quant_module import qconfig_mapping


class _Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self._impl = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self._impl(x)
        return x


class _ConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self._impl = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self._impl(x)
        return x


class QuantConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.submodule = _Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.submodule = prepare_fx(
            self.submodule,
            qconfig_mapping,
            torch.randn(1, in_channels, 32, 32),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.submodule(x)
        return x


class QuantConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.submodule = _ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
        )
        self.submodule = prepare_fx(
            self.submodule,
            qconfig_mapping,
            torch.randn(1, in_channels, 32, 32),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.submodule(x)
        return x


class _ConvNxN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.eps = eps
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.norm = nn.InstanceNorm2d(out_channels, eps=eps)
        self.act = nn.SiLU(True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)

        x = self.act(x)

        return x


class _QuantConvNxN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.submodule = _ConvNxN(
            in_channels, out_channels, kernel_size, stride, padding, eps
        )
        self.submodule = prepare_fx(
            self.submodule,
            qconfig_mapping,
            torch.randn(1, in_channels, 32, 32),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.submodule(x)
        return x


class Conv3x3(_ConvNxN):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, eps: float = 1e-5
    ):
        super().__init__(in_channels, out_channels, 3, stride, 1, eps)


class QuantConv3x3(_QuantConvNxN):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, eps: float = 1e-5
    ):
        super().__init__(in_channels, out_channels, 3, stride, 1, eps)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, eps: float = 1e-5):
        super().__init__()
        self.conv1 = QuantConv3x3(in_channels, out_channels, eps=eps)
        self.conv2 = QuantConv3x3(out_channels, out_channels, eps=eps)
        self.conv3 = QuantConv2d(out_channels, out_channels, 1, bias=False)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else QuantConv2d(in_channels, out_channels, 1, bias=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.shortcut(x) + self.conv3(self.conv2(self.conv1(x)))
        return x
