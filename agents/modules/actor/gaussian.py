from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Sequence

import torch
from torch import nn
from torch.distributions import Normal


class GaussianActor(nn.Module):
    """连续有界高斯策略"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_min: torch.Tensor,
        action_max: torch.Tensor,
        hidden_sizes: Sequence[int] = (),
    ):
        super().__init__()

        from ..utils import MLP, init_net

        self._net = MLP(state_dim, action_dim + action_dim, hidden_sizes)

        init_net(self._net, init_type="orthogonal", init_gain=0.01)

    def get_kern(self, state: torch.Tensor):
        r"""获取[0,1]上的高斯分布参数(\mu, \sigma)"""
        mu, std = torch.chunk(self._net(state), 2, -1)
        mu = torch.sigmoid(mu)  # Ensure the output is in [0, 1]
        std = torch.sigmoid(std) * (1 / 6)  # Ensure the std is in [0, 1/6]
        return mu, std

    def forward(self, state: torch.Tensor):
        mu, std = self.get_kern(state)
        return mu

    def get_dist(self, state: torch.Tensor) -> Normal:
        mu_k, std_k = self.get_kern(state)
        dist = Normal(mu_k, std_k)
        return dist
