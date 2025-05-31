import torch
import math
from typing import TYPE_CHECKING, Sequence
from environments.utils.math import euler_from_quat
from .base_reward_fn import BaseRewardFn

if TYPE_CHECKING:
    from environments import NavigationEnv, EvasionEnv


class TimeRewardFn(BaseRewardFn):
    def __init__(self, weight: float = 1) -> None:
        super().__init__()
        self.weight = weight

    def reset(self, env: "NavigationEnv | EvasionEnv", env_indices: Sequence[int] | torch.Tensor | None = None):
        pass
        
    def __call__(self, env: "NavigationEnv | EvasionEnv", **kwargs) -> torch.Tensor:
        return self.weight*torch.ones((env.num_envs, 1), device=env.device)