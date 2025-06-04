from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..navigation import NavigationEnv
    from ..evasion import EvasionEnv
    from .base_reward_fn import _EnvIndexType
import torch
from typing import Sequence
from .base_reward_fn import BaseRewardFn


if TYPE_CHECKING:
    from ..navigation import NavigationEnv


class LowAltitudeRewardFn(BaseRewardFn):
    """高度过低惩罚"""

    def __init__(self, min_altitude_m: float, weight: float = 1, version=2) -> None:
        super().__init__()
        self.min_altitude_m = min_altitude_m
        self.weight = weight
        self.version = version

    def reset(self, env: "NavigationEnv", env_indices: _EnvIndexType = None):
        pass

    def forward(self, env: "NavigationEnv", plane, **kwargs) -> torch.Tensor:
        h = plane.altitude_m()
        if self.version == 1:
            yes = h < self.min_altitude_m
            rew = -yes
        elif self.version == 2:  # 对应约束 h_min-h<=0
            rew = h - self.min_altitude_m
        else:
            raise NotImplementedError(self)
        return rew
