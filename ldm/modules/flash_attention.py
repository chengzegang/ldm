# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.ao.quantization.quantize_fx import prepare_fx
from xformers.ops import memory_efficient_attention

from .quant_module import qconfig_mapping


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        head_size: int,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.head_size = head_size
        self.norm = nn.LayerNorm(embed_dim)
        self.in_proj = nn.Linear(embed_dim * 3, embed_dim * 3)

    def _split_heads(self, x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], x.shape[1], self.embed_dim // self.head_size, -1)

    def _merge_heads(self, x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], x.shape[1], -1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.in_proj(x.repeat(1, 1, 3))
        qkv = self._split_heads(x)
        q, k, v = qkv.chunk(3, dim=-1)
        if torch.jit.is_scripting() or not torch.jit.is_tracing():
            a_val = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
            ).transpose(1, 2)
        else:
            a_val = memory_efficient_attention(q, k, v)
        x = self._merge_heads(a_val)
        return x


class QuantMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        head_size: int,
    ) -> None:
        super().__init__()
        self.submodule = MultiheadAttention(embed_dim, head_size)
        self.submodule = prepare_fx(
            self.submodule,
            qconfig_mapping,
            torch.randn(1, 64, embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.submodule(x)


class MultiheadCrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        head_size: int,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.head_size = head_size
        self.norm = nn.LayerNorm(embed_dim)
        self.in_proj = nn.Linear(embed_dim, embed_dim)

    def _split_heads(self, x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], x.shape[1], self.embed_dim // self.head_size, -1)

    def _merge_heads(self, x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], x.shape[1], -1)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = self.norm(x)
        y = self.norm(y)
        q = self.in_proj(x)
        kv = self.in_proj(y)
        q = self._split_heads(q)
        kv = self._split_heads(kv)
        if torch.jit.is_scripting() or not torch.jit.is_tracing():
            a_val = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                kv.transpose(1, 2),
                kv.transpose(1, 2),
            ).transpose(1, 2)
        else:
            a_val = memory_efficient_attention(q, kv, kv)
        x = self._merge_heads(a_val)
        return x


class QuantMultiheadCrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        head_size: int,
    ) -> None:
        super().__init__()
        self.submodule = MultiheadCrossAttention(embed_dim, head_size)
        self.submodule = prepare_fx(
            self.submodule,
            qconfig_mapping,
            torch.randn(1, 1024, embed_dim),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.submodule(x, y)
