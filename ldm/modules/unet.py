from typing import List, Optional

from torch import Tensor, nn
from .attention import AttentionLayer2d, QuantCrossAttentionLayer2d
from .convolutions import ResidualBlock, QuantConvTranspose2d, QuantConv2d
import torch
from torch import fx


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
        hidden_size: int,
        latent_size: int,
        eps: float = 1e-4,
        num_transformer_layers: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.latent_size = latent_size
        self.layers = nn.ModuleList()
        self.in_conv = QuantConv2d(in_channels, channels[0], kernel_size=1)
        for i in range(len(channels) - 1):
            self.layers.append(ConvDown(channels[i], channels[i + 1], eps))
        self.layers.append(ResidualBlock(channels[-1], hidden_size, eps))
        self.attn = nn.ModuleList(
            [
                QuantCrossAttentionLayer2d(
                    hidden_size,
                    128,
                )
                for _ in range(num_transformer_layers)
            ]
        )
        self.out_conv = QuantConv2d(hidden_size, latent_size, kernel_size=1)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        x = self.in_conv(x)
        for layer in self.layers:
            x = layer(x)
        for attn in self.attn:
            x = attn(x, cond)
        z = self.out_conv(x)
        return z


class UNetDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        channels: List[int],
        hidden_size: int,
        latent_size: int,
        eps: float = 1e-4,
        num_transformer_layers: int = 1,
    ):
        super().__init__()
        self.out_channels = out_channels
        channels = channels[::-1]
        self.in_conv = QuantConv2d(latent_size, hidden_size, kernel_size=1)
        self.attn = nn.ModuleList(
            [
                QuantCrossAttentionLayer2d(
                    hidden_size,
                    128,
                )
                for _ in range(num_transformer_layers)
            ]
        )
        self.layers = nn.ModuleList()
        self.layers.append(ResidualBlock(hidden_size, channels[0], eps))
        for i in range(len(channels) - 1):
            self.layers.append(ConvUp(channels[i], channels[i + 1], eps))
        self.out_norm = nn.InstanceNorm2d(channels[-1], eps=eps)
        self.out = QuantConv2d(channels[-1], out_channels, kernel_size=1)

    def forward(self, qz: Tensor, cond: Tensor) -> Tensor:
        x = self.in_conv(qz)
        for attn in self.attn:
            x = attn(x, cond)
        for layer in self.layers:
            x = layer(x)
        x = self.out(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        hidden_size,
        latent_size: int,
        num_transformer_layers: int = 1,
    ):
        super().__init__()
        self.encoder = UNetEncoder(
            in_channels,
            channels,
            hidden_size,
            latent_size,
            num_transformer_layers=num_transformer_layers,
        )
        self.decoder = UNetDecoder(
            in_channels,
            channels,
            hidden_size,
            latent_size,
            num_transformer_layers=num_transformer_layers,
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        return self.decoder(self.encoder(x, cond), cond)

    @classmethod
    def apply_shortcuts(cls, unet: "UNet") -> "UNet":
        g = fx.symbolic_trace(unet)
        down_ops = []
        up_ops = []
        for node in g.graph.nodes:
            if "down" in node.name:
                down_ops.append(node)
        for node in g.graph.nodes:
            if "up" in node.name:
                up_ops.append(node)

        for dop, uop in zip(down_ops, up_ops[::-1]):
            dop.args[0]
            with g.graph.inserting_after(dop):
                out_node = g.graph.create_node(
                    "call_function", torch.clone, args=(dop,), name=dop.name + "_clone"
                )
            with g.graph.inserting_before(uop):
                in_node = g.graph.create_node(
                    "call_function",
                    torch.add,
                    args=(out_node, uop.args[0]),
                    name=uop.name + "_add",
                )
            uop.args = (in_node,)

        g.graph.lint()

        g.recompile()
        return g
