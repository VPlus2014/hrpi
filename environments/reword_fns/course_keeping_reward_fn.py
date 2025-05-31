import torch
import math
from typing import TYPE_CHECKING, Sequence
from environments.utils.math import euler_from_quat
from .base_reward_fn import BaseRewardFn

if TYPE_CHECKING:
    from environments.navigation import NavigationEnv


class CourseKeepingRewardFn(BaseRewardFn):
    def __init__(self, course: float, weight: float = 1) -> None:
        super().__init__()
        self.course = course
        self.weight = weight

    def reset(self, env: "NavigationEnv", env_indices: Sequence[int] | torch.Tensor | None = None):
        pass
        
    def __call__(self, env: "NavigationEnv", **kwargs) -> torch.Tensor:
        euler = euler_from_quat(env.aircraft.q_kg)
        chi = euler[..., 2:3]
        err = torch.abs(math.sin(self.course)-torch.sin(chi)) + torch.abs(math.cos(self.course)-torch.cos(chi))
        
        return -1*self.weight*err