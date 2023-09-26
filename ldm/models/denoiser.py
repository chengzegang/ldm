import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union
from ..modules.attention import QuantCrossAttentionLayer


class Transformer2d(nn.Module):
    def __init__(
        self,
        latent_size: int,
        hidden_size: int,
        num_layers: int,
        clip_emb_size: int,
        attn_head_size: int = 128,
    ):
        super().__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.clip_emb_size = clip_emb_size
        self.layers = nn.ModuleList()
        self.latent_emb = nn.Linear(latent_size * 2, hidden_size)
        self.clip_emb = nn.Linear(clip_emb_size, hidden_size)
        self.time_emb = nn.Embedding(10000, hidden_size)
        for i in range(num_layers):
            self.layers.append(QuantCrossAttentionLayer(hidden_size, attn_head_size))
        self.latent_out_proj = nn.Linear(hidden_size, latent_size)

    def forward(
        self, latent: Tensor, mask: Tensor, cond_emb: Tensor, t_emb: Tensor
    ) -> Tensor:
        shape = latent.shape
        latent = torch.cat([latent, mask], dim=1).flatten(2).transpose(-1, -2)
        latent_emb = self.latent_emb(latent)
        cond_emb = self.clip_emb(cond_emb)
        t_emb = self.time_emb(t_emb).unsqueeze(1)
        cond_emb = torch.cat([cond_emb, t_emb], dim=1)
        for layer in self.layers:
            latent_emb = layer(latent_emb, cond_emb)
        latent = self.latent_out_proj(latent_emb)
        latent = latent.transpose(-1, -2).view(shape)
        return latent
