from __future__ import annotations
from typing import TYPE_CHECKING

# import torch
from .base_reward_fn import BaseRewardFn


class LowAirSpeedRewardFn(BaseRewardFn):
    def __init__(
        self,
        min_airspeed_mps: float,
        max_airspeed_mps: float,
        weight: float = 1.0,
        version=3,
    ) -> None:
        super().__init__()
        self.min_airspeed_mps = min_airspeed_mps
        self.max_airspeed_mps = max_airspeed_mps
        assert self.min_airspeed_mps < self.max_airspeed_mps
        self._vinvspan = 1 / (self.max_airspeed_mps - self.min_airspeed_mps)
        self.weight = weight
        self.version = version

    def reset(self, env, env_indices=None, **kwargs):
        pass

    def forward(self, env, plane, **kwargs) -> torch.Tensor:
        tas = plane.tas()
        vtld = (tas - self.min_airspeed_mps) * self._vinvspan
        if self.version == 1:  # 事件型
            reward = -((vtld > 0).to(env.dtype))
        elif self.version == 2:  # 线性型
            reward = vtld
        elif self.version == 3:  # 障碍函数
            eps = 1e-2
            reward = 1 / (-eps - vtld)
        else:
            raise NotImplementedError
        return reward
