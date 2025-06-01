from __future__ import annotations
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from typing import Sequence
    from torch.distributions import Distribution

import torch
from torch import nn
from .discrete_normal import IndependentDiscreteDistribution


class GaussianDActor(nn.Module):
    """连续有界离散高斯策略"""

    def __init__(
        self,
        state_dim: int,
        action_nvec: Sequence[int],  # 各维度的离散化大小
        action_min: torch.Tensor,
        action_max: torch.Tensor,
        hidden_sizes: Sequence[int] = (),
        joint: bool = False,
    ):
        super().__init__()

        self.action_nvec = action_nvec = tuple([*action_nvec])
        self.action_dim = action_dim = len(action_nvec)
        self._nvec_tsr = torch.tensor(action_nvec, dtype=torch.int64)

        assert (
            action_min.shape == action_max.shape
        ), "action_min and action_max must have the same shape"
        assert action_min.shape[-1] == len(action_nvec), (
            f"expected action_min.shape[-1] == {action_dim}, got",
            action_min.shape[-1],
        )
        assert self._nvec_tsr.min() >= 1, "action_nvec must be greater than 1"
        self.action_min = action_min
        self.action_max = action_max
        self.action_range = action_max - action_min
        self._r_d2c = torch.where(
            self._nvec_tsr > 1,
            self.action_range / (self._nvec_tsr - 1),
            torch.zeros_like(self.action_range),
        )
        self._r_c2d = torch.where(
            self.action_range != 0,
            (self._nvec_tsr - 1) / self.action_range,
            torch.zeros_like(self.action_range),
        )
        self.register_buffer("_action_min", self.action_min)
        self.register_buffer("_action_max", self.action_max)
        self.register_buffer("_action_range", self.action_range)
        self.register_buffer("_r_d2c", self._r_d2c)
        self.register_buffer("_r_c2d", self._r_c2d)
        self.joint = joint

        from ..utils import MLP, init_net

        self._mlp = MLP(state_dim, action_dim * 2, hidden_sizes)

        init_net(self._mlp, init_type="orthogonal", init_gain=0.01)

    def forward(self, state: torch.Tensor):
        mu, std = torch.chunk(self._mlp(state), 2, dim=-1)
        mu = torch.sigmoid(mu)  # in (0,1)
        std = torch.sigmoid(std).clamp(min=1e-6)  # in (1e-6, 1)
        return mu, std

    def get_dist(self, state: torch.Tensor) -> IndependentDiscreteDistribution:
        """获取离散分布"""
        mu, std = self.forward(state)
        dist = IndependentDiscreteDistribution(
            mu, std, nvec=self.action_nvec, joint=self.joint
        )
        return dist

    def d2c(self, action_d: torch.LongTensor) -> torch.Tensor:
        """将离散动作转换为连续动作"""
        action_c = self.action_min.expand_as(action_d) + action_d / (
            self._r_d2c.expand_as(action_d)
        )
        return action_c

    def c2d(self, action_c: torch.Tensor) -> torch.LongTensor:
        """将连续动作转换为离散动作"""
        action_d = (
            (
                (action_c - self.action_min.expand_as(action_c))
                * self._r_c2d.expand_as(action_c)
            )
        ).to(torch.int64)
        action_d = torch.clamp(
            action_d, torch.zeros_like(self._nvec_tsr), self._nvec_tsr - 1
        )
        action_d = cast(torch.LongTensor, action_d)
        return action_d
