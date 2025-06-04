from __future__ import annotations
from typing import TYPE_CHECKING

import torch
from .base_reward_fn import BaseRewardFn


class LowAirSpeedRewardFn(BaseRewardFn):
    def __init__(self, min_airspeed_mps: float, weight: float = 1.0, version=1) -> None:
        super().__init__()
        self.min_airspeed_mps = min_airspeed_mps
        self.weight = weight
        self.version = version

    def reset(self, env, env_indices=None, **kwargs):
        pass

    def forward(self, env, plane, **kwargs) -> torch.Tensor:
        tas = plane.tas()
        if self.version == 1:
            reward = -(tas < self.min_airspeed_mps)
        elif self.version == 2:
            reward = tas - self.min_airspeed_mps
        else:
            raise NotImplementedError
        return reward
