from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..navigation import NavigationEnv
    from ..evasion import EvasionEnv
    from .base_reward_fn import IndexLike
import torch
from .base_reward_fn import BaseRewardFn

if TYPE_CHECKING:
    from ..navigation import NavigationEnv


class ReachNavigationPointRewardFn(BaseRewardFn):
    def __init__(self, min_distance_m: float, weight: float = 1) -> None:
        super().__init__()
        self.min_distance_m = min_distance_m
        self.weight = weight

    def reset(self, env: "NavigationEnv", env_indices: IndexLike | None = None):
        pass

    def __call__(self, env: "NavigationEnv", **kwargs) -> torch.Tensor:
        reward = torch.zeros(size=(env.num_envs, 1), device=env.device)
        selected_points = env.current_navigation_point
        distance = torch.norm(
            env.aircraft.position_g - selected_points, dim=-1, p=2, keepdim=True
        ).detach()
        reached = distance <= self.min_distance_m
        if torch.any(reached):
            indices = torch.where(reached)[0]
            env.navigation_point_index[indices] += 1
            reward[indices] += 1
            for reward_fn in env._reward_fns:
                reward_fn.reset(env, indices)
            if 0 in indices:
                # render
                env.render_navigation_points()
        return self.weight * reward
