from __future__ import annotations
from typing import TYPE_CHECKING

# import torch
from .base_reward_fn import BaseRewardFn


class AltitudeKeepingRewardFn(BaseRewardFn):
    def __init__(self, altitude: float, weight: float = 1) -> None:
        super().__init__()
        self.altitude = altitude
        self.weight = weight

    def reset(self, env, env_indices=None, **kwargs):
        pass

    def forward(self, env, plane, **kwargs) -> torch.Tensor:
        err = torch.abs(self.altitude - unit.altitude())
        return -err
