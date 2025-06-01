from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..navigation import NavigationEnv
    from ..evasion import EvasionEnv
    from .base_reward_fn import IndexLike
import torch
from .base_reward_fn import BaseRewardFn


class LowAirSpeedRewardFn(BaseRewardFn):
    def __init__(self, min_airspeed_mps: float, weight: float = 1) -> None:
        super().__init__()
        self.min_airspeed_mps = min_airspeed_mps
        self.weight = weight

    def reset(self, env: "NavigationEnv", env_indices: IndexLike | None = None):
        pass

    def __call__(self, env: "NavigationEnv", **kwargs) -> torch.Tensor:
        reward = (env.aircraft.tas < self.min_airspeed_mps).detach()
        return -1 * self.weight * reward
