from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Sequence, Literal

import torch
import torch.nn as nn


def init_net(net: nn.Module, init_type: str = "orthogonal", init_gain: float = 1.0):
    """Initialize a network"""

    def init_func(m: nn.Module):
        if isinstance(m, nn.Linear):
            if init_type == "normal":
                nn.init.normal_(m.weight, 0.0, init_gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight, gain=init_gain)
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight, gain=init_gain)
            else:
                raise NotImplementedError(
                    f"Initialization type {init_type} is not implemented."
                )

            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    net.apply(init_func)


# Trick 8: orthogonal initialization
def orthogonal_init(layer: nn.Linear, gain=1.0):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


class MLP(nn.Module):
    r"""
    \Phi(x) = b + \sum_{i} w_i \phi_i(x)
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        hidden_sizes: Sequence[int] = (),
        activation: Literal["relu", "tanh", "sigmoid"] = "relu",
    ):
        super().__init__()

        layers = []
        hidden_sizes = [dim_in, *list(hidden_sizes)]

        if activation == "relu":
            actv = nn.ReLU()
        elif activation == "tanh":
            actv = nn.Tanh()
        elif activation == "sigmoid":
            actv = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        for in_dim, out_dim in zip(hidden_sizes[:-1], hidden_sizes[1:], strict=True):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(actv)

        layers.append(nn.Linear(hidden_sizes[-1], dim_out))

        self._net = nn.Sequential(*layers)

        init_net(self._net, init_type="orthogonal")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y: torch.Tensor = self._net(x)
        return y


class ResMLP(nn.Module):
    """线性+非线性映射"""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        hidden_sizes: Sequence[int] = (),
        activation: Literal["relu", "tanh", "sigmoid"] = "relu",
    ):
        super().__init__()

        self._nlin = MLP(dim_in, dim_out, hidden_sizes, activation=activation)
        self._lin = (
            nn.Linear(dim_in, dim_out, bias=False)
            if dim_in != dim_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._nlin(x) + self._lin(x)
