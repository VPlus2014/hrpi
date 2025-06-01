from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Sequence, Literal
import torch
import torch.nn as nn

from .utils import MLP


class Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_sizes: Sequence[int] = (),
    ):
        super().__init__()

        self._net = MLP(state_dim, 1, hidden_sizes)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        v: torch.Tensor = self._net(state)
        return v
