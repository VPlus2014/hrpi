import torch
import math
from typing import TYPE_CHECKING, Sequence
from environments.utils.math import euler_from_quat
from .base_reward_fn import BaseRewardFn

if TYPE_CHECKING:
    from environments.navigation import NavigationEnv


class AltitudeKeepingRewardFn(BaseRewardFn):
    def __init__(self, altitude: float, weight: float = 1) -> None:
        super().__init__()
        self.altitude = altitude
        self.weight = weight

    def reset(self, env: "NavigationEnv", env_indices: Sequence[int] | torch.Tensor | None = None):
        pass
        
    def __call__(self, env: "NavigationEnv", **kwargs) -> torch.Tensor:
        err = torch.abs(self.altitude-env.aircraft.altitude_m)
        
        return -1*self.weight*err