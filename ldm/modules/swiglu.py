from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.ao.quantization.quantize_fx import prepare_fx
from xformers.ops.swiglu_op import swiglu
from .quant_module import qconfig_mapping


class SwiGLU(nn.Module):
    """
    A Module that encapsulates the call to :attr:`xformers.ops.swiglu`,
    and holds the weights for the 3 linear layers
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        bias: bool = True,
    ) -> None:
        """Create a SwiGLU module

        Args:
            in_features (int): Number of features of the input
            hidden_features (int): Number of hidden features
            out_features (Optional[int], optional): Number of features of the input. Defaults to None.
            bias (bool, optional): Whether linear layers also include a bias. Defaults to True.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

        self.hidden_features = hidden_features
        self.out_features = out_features
        self.in_features = in_features

    def forward(self, x: Tensor) -> Tensor:
        """Computes :attr:`swiglu` with the module's weights

        Args:
            x (Tensor): A Tensor of shape ``[..., in_features]``

        Returns:
            Tensor: A Tensor of shape ``[..., out_features]``
        """
        if torch.jit.is_scripting() or not torch.jit.is_tracing():
            x1 = self.w1(x)
            x2 = self.w2(x)
            hidden = F.silu(x1, True) * x2
            return self.w3(hidden)
        else:
            return swiglu(
                x,
                self.w1.weight,
                self.w1.bias,
                self.w2.weight,
                self.w2.bias,
                self.w3.weight,
                self.w3.bias,
            )


class QuantSwiGLU(nn.Module):
    """
    A Module that encapsulates the call to :attr:`xformers.ops.swiglu`,
    and holds the weights for the 3 linear layers
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.submodule = SwiGLU(in_features, hidden_features, out_features, bias)
        self.submodule = prepare_fx(
            self.submodule,
            qconfig_mapping,
            torch.randn(1, 64, in_features),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.submodule(x)
