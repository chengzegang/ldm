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
        cond_size: int,
        out_channels: int,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.proj = nn.Linear(cond_size, in_channels)
        self.attn = QuantCrossAttentionLayer2d(in_channels, 64)
        self.res = ResidualBlock(in_channels, out_channels, eps)
        self.down = QuantConv2d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        cond = self.proj(cond)
        x = self.attn(x, cond)
        x = self.res(x)
        x = self.down(x)
        return x


class ConvUp(nn.Module):
    def __init__(
        self,
        in_channels: int,
        cond_size: int,
        out_channels: int,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.proj = nn.Linear(cond_size, in_channels)
        self.merge = QuantConv2d(in_channels * 2, in_channels, kernel_size=1)
        self.attn = QuantCrossAttentionLayer2d(in_channels, 64)
        self.up = QuantConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.res = ResidualBlock(out_channels, out_channels, eps)

    def forward(self, x: Tensor, y: Tensor, cond: Tensor) -> Tensor:
        x = self.merge(torch.cat([x, y], dim=1))
        cond = self.proj(cond)
        x = self.attn(x, cond)
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
            self.layers.append(ConvDown(channels[i], hidden_size, channels[i + 1], eps))
        self.pre_attn_conv = QuantConv2d(channels[-1], hidden_size, kernel_size=1)
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

    def forward(self, x: Tensor, cond: Tensor) -> List[Tensor]:
        x = self.in_conv(x)
        xs = [x]
        for layer in self.layers:
            xs.append(layer(xs[-1], cond))
        last = xs[-1]
        last = self.pre_attn_conv(last)
        for attn in self.attn:
            last = attn(last, cond)
        last = self.out_conv(last)
        xs.append(last)
        return xs


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
        self.post_attn_conv = QuantConv2d(hidden_size, channels[0], kernel_size=1)
        for i in range(len(channels) - 1):
            self.layers.append(ConvUp(channels[i], hidden_size, channels[i + 1], eps))
        self.out_norm = nn.InstanceNorm2d(channels[-1], eps=eps)
        self.out = nn.Conv2d(channels[-1], out_channels, kernel_size=1)

    def forward(self, xs: Tensor, cond: Tensor) -> Tensor:
        xs[-1] = self.in_conv(xs[-1])
        for attn in self.attn:
            xs[-1] = attn(xs[-1], cond)
        x = xs.pop(-1)
        x = self.post_attn_conv(x)
        for layer, y in zip(self.layers, xs[::-1]):
            x = layer(x, y, cond)
        x = self.out(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
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
            out_channels,
            channels,
            hidden_size,
            latent_size,
            num_transformer_layers=num_transformer_layers,
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        return self.decoder(self.encoder(x, cond), cond)


# @classmethod
# def apply_shortcuts(cls, unet: "UNet") -> "UNet":
#    g = fx.symbolic_trace(unet)
#    down_ops = []
#    up_ops = []
#    for node in g.graph.nodes:
#        if "down" in node.name:
#            down_ops.append(node)
#    for node in g.graph.nodes:
#        if "up" in node.name:
#            up_ops.append(node)

#    for dop, uop in zip(down_ops, up_ops[::-1]):
#        dop.args[0]
#        with g.graph.inserting_after(dop):
#            out_node = g.graph.create_node(
#                "call_function", torch.clone, args=(dop,), name=dop.name + "_clone"
#            )
#        with g.graph.inserting_before(uop):
#            in_node = g.graph.create_node(
#                "call_function",
#                torch.add,
#                args=(out_node, uop.args[0]),
#                name=uop.name + "_add",
#            )
#        uop.args = (in_node,)

#    g.graph.lint()

#    g.recompile()
#    return g
