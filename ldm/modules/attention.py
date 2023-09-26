from .flash_attention import (
    MultiheadCrossAttention,
    QuantMultiheadCrossAttention,
    QuantMultiheadAttention,
    MultiheadAttention,
)
from .swiglu import SwiGLU, QuantSwiGLU
import torch
from torch import nn, Tensor


class CrossAttentionLayer(nn.Module):
    def __init__(self, model_dim: int, head_size: int):
        super().__init__()
        self.sa = MultiheadCrossAttention(model_dim, head_size)
        self.sa_mlp = SwiGLU(model_dim, model_dim * 4)
        self.ca = MultiheadCrossAttention(model_dim, head_size)
        self.ca_mlp = SwiGLU(model_dim, model_dim * 4)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = x + self.sa(x, x)
        x = x + self.sa_mlp(x)
        x = x + self.ca(x, y)
        x = x + self.ca_mlp(x)

        return x


class QuantCrossAttentionLayer(nn.Module):
    def __init__(self, model_dim: int, head_size: int):
        super().__init__()
        self.sa = QuantMultiheadCrossAttention(model_dim, head_size)
        self.sa_mlp = QuantSwiGLU(model_dim, model_dim * 4)
        self.ca = QuantMultiheadCrossAttention(model_dim, head_size)
        self.ca_mlp = QuantSwiGLU(model_dim, model_dim * 4)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = x + self.sa(x, x)
        x = x + self.sa_mlp(x)
        x = x + self.ca(x, y)
        x = x + self.ca_mlp(x)

        return x


class AttentionLayer(nn.Module):
    def __init__(self, model_dim: int, head_size: int):
        super().__init__()
        self.attn = MultiheadAttention(model_dim, head_size)
        self.mlp = SwiGLU(model_dim, model_dim * 4)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x


class QuantAttentionLayer(nn.Module):
    def __init__(self, model_dim: int, head_size: int):
        super().__init__()
        self.attn = QuantMultiheadAttention(model_dim, head_size)
        self.mlp = QuantSwiGLU(model_dim, model_dim * 4)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x


class AttentionLayer2d(AttentionLayer):
    def forward(self, x: Tensor) -> Tensor:
        return (
            super().forward(x.flatten(2).transpose(-1, -2)).transpose(-1, -2).view_as(x)
        )


class QuantAttentionLayer2d(QuantAttentionLayer):
    def forward(self, x: Tensor) -> Tensor:
        return (
            super().forward(x.flatten(2).transpose(-1, -2)).transpose(-1, -2).view_as(x)
        )


class CrossAttentionLayer2d(CrossAttentionLayer):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return (
            super()
            .forward(x.flatten(2).transpose(-1, -2), y)
            .transpose(-1, -2)
            .view_as(x)
        )


class QuantCrossAttentionLayer2d(QuantCrossAttentionLayer):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return (
            super()
            .forward(x.flatten(2).transpose(-1, -2), y)
            .transpose(-1, -2)
            .view_as(x)
        )
