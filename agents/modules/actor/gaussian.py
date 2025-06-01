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

        assert action_min.shape == action_max.shape
        self.action_min = action_min
        self.action_max = action_max
        self.action_range = action_max - action_min
        self.register_buffer("_action_min", self.action_min)
        self.register_buffer("_action_max", self.action_max)
        self.register_buffer("_action_range", self.action_range)

        from ..utils import MLP, init_net

        self._net = MLP(state_dim, action_dim, hidden_sizes)

        init_net(self._net, init_type="orthogonal", init_gain=0.01)

        self.log_std = nn.Parameter(-2 * torch.ones(1, action_dim))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        mu = self._net(state)  # 预期输出均值为0
        mu = torch.sigmoid(mu)  # Ensure the output is in [0, 1]
        return mu

    def get_dist(self, state: torch.Tensor) -> Normal:
        mu = self.forward(state)  # in [0,1]
        mean: torch.Tensor = self.action_min + self.action_range * mu

        log_std = self.log_std.expand_as(mu)
        std = torch.exp(log_std).clamp(min=1e-6)

        dist = Normal(mean, std)
        return dist
