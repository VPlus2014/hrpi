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
    from environments import NavigationEnv, EvasionEnv


class TimeRewardFn(BaseRewardFn):
    def __init__(self, weight: float = 1) -> None:
        super().__init__()
        self.weight = weight

    def reset(
        self, env: "NavigationEnv | EvasionEnv", env_indices: IndexLike | None = None
    ):
        pass

    def __call__(self, env: "NavigationEnv | EvasionEnv", **kwargs) -> torch.Tensor:
        return self.weight * torch.ones((env.num_envs, 1), device=env.device)
