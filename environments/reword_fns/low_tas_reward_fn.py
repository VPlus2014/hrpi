import torch
from typing import TYPE_CHECKING, Sequence
from .base_reward_fn import BaseRewardFn

if TYPE_CHECKING:
    from environments.navigation import NavigationEnv


class LowAirSpeedRewardFn(BaseRewardFn):
    def __init__(self, min_airspeed_mps: float, weight: float = 1) -> None:
        super().__init__()
        self.min_airspeed_mps = min_airspeed_mps
        self.weight = weight

    def reset(self, env: "NavigationEnv", env_indices: Sequence[int] | torch.Tensor | None = None):
        pass
        
    def __call__(self, env: "NavigationEnv", **kwargs) -> torch.Tensor:
        reward = (env.aircraft.tas < self.min_airspeed_mps).detach().to(torch.float32)
        return -1*self.weight*reward