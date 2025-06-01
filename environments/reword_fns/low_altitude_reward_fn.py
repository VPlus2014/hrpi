from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..navigation import NavigationEnv
    from ..evasion import EvasionEnv
    from .base_reward_fn import IndexLike
import torch
from typing import Sequence
from .base_reward_fn import BaseRewardFn


if TYPE_CHECKING:
    from ..navigation import NavigationEnv


class LowAltitudeRewardFn(BaseRewardFn):
    def __init__(self, min_altitude_m: float, weight: float = 1) -> None:
        super().__init__()
        self.min_altitude_m = min_altitude_m
        self.weight = weight

    def reset(self, env: "NavigationEnv", env_indices: IndexLike | None = None):
        pass

    def __call__(self, env: "NavigationEnv", **kwargs) -> torch.Tensor:
        reward = (
            (env.aircraft.altitude_m < self.min_altitude_m).detach()
        )
        return -1 * self.weight * reward
