import torch
from typing import TYPE_CHECKING, Sequence
from .base_reward_fn import BaseRewardFn

if TYPE_CHECKING:
    from environments.navigation import NavigationEnv


class LowAltitudeRewardFn(BaseRewardFn):
    def __init__(self, min_altitude_m: float, weight: float = 1) -> None:
        super().__init__()
        self.min_altitude_m = min_altitude_m
        self.weight = weight

    def reset(self, env: "NavigationEnv", env_indices: Sequence[int] | torch.Tensor | None = None):
        pass
        
    def __call__(self, env: "NavigationEnv", **kwargs) -> torch.Tensor:
        reward = (env.aircraft.altitude_m < self.min_altitude_m).detach().to(torch.float32)
        return -1*self.weight*reward