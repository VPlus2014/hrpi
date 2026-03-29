from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, List, TypeVar, Union, Sequence

if TYPE_CHECKING:
    from typing import Literal


import numpy as np
import torch
import torch.nn as nn
from tianshou.utils.net.common import MLP as _MLP, NetBase as _NetBase, TRecurrentState
from .cvrt import to_torch, NDArray

_T = TypeVar("_T")
_Kleene = Union[Sequence[_T], _T]
ModuleType = type[nn.Module]
ArgsType = Union[
    _Kleene[tuple[Any, ...]],
    _Kleene[dict[Any, Any]],
]
TActionShape = Union[Sequence[int], int, np.int64]
TLinearLayer = Callable[[int, int], nn.Module]
DeviceLike = Union[str, int, torch.device]


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
def orthogonal_init_(layer: nn.Module, gain=1.0):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


def parse_activation(activation: str) -> nn.Module:
    if activation == "relu":
        return nn.ReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unsupported activation function: {activation}")


class NNModule(nn.Module):
    def to(
        self,
        device: DeviceLike | None = None,
        dtype: torch.dtype | None = None,
        non_blocking=False,
    ):
        if device is not None:
            self._device = torch.device(device)
        if dtype is not None:
            self._dtype = dtype
        return super().to(device, dtype, non_blocking=non_blocking)
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    @property
    def dtype(self) -> torch.dtype:
        return self._dtype


class MLP(NNModule):
    r"""
    \Phi(x) = b + \sum_{i} w_i \phi_i(x)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 0,
        hidden_sizes: Sequence[int] = (),
        norm_layer: _Kleene[ModuleType] | None = None,
        norm_args: ArgsType | None = None,
        activation: _Kleene[ModuleType] | None = nn.ReLU,
        act_args: ArgsType | None = None,
        linear_layer: TLinearLayer = nn.Linear,
        device: DeviceLike = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.__doc__ = _MLP.__doc__
        super().__init__()
        self._kern = _MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            norm_layer=norm_layer,
            norm_args=norm_args,
            activation=activation,
            act_args=act_args,
            device=device,
            linear_layer=linear_layer,
            flatten_input=False,
        )
        self.to(device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._kern.model(x)
        return y


class ResMLP(NNModule):
    """Linear Residual MLP"""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        nonlinear: MLP,
        linear_with_identity=True,
        device: DeviceLike = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self._nlin = nonlinear
        self._lin = (
            nn.Identity()
            if dim_in == dim_out and linear_with_identity
            else nn.Linear(dim_in, dim_out, bias=False)
        )
        self.input_dim = dim_in
        self.output_dim = dim_out
        self.to(device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._nlin(x) + self._lin(x)


class NetBase(NNModule, _NetBase[TRecurrentState]):
    r"""infer input with obs & hidden state(optional), \
        output with action&next hidden state(optional)"""

    pass


class MLP_NoState(NetBase[None]):

    def __init__(
        self,
        model: NNModule,
    ):
        super().__init__()
        self.model = model
        self.to(model.device, model.dtype)

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state=None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, None]:
        obs = torch.as_tensor(obs, device=self.device, dtype=self.dtype)
        y = self.model(obs)
        return y, None
