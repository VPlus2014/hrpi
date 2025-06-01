from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..navigation import NavigationEnv
    from ..evasion import EvasionEnv
    from .base_reward_fn import IndexLike


import torch
import math
from environments.utils.math import euler_from_quat
from .base_reward_fn import BaseRewardFn

if TYPE_CHECKING:
    from environments.navigation import NavigationEnv


class AltitudeKeepingRewardFn(BaseRewardFn):
    def __init__(self, altitude: float, weight: float = 1) -> None:
        super().__init__()
        self.altitude = altitude
        self.weight = weight

    def reset(self, env: "NavigationEnv", env_indices: IndexLike|None = None):
        pass
        
    def __call__(self, env: "NavigationEnv", **kwargs) -> torch.Tensor:
        err = torch.abs(self.altitude-env.aircraft.altitude_m)
        
        return -1*self.weight*err
