from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...navigation import NavigationEnv
# import torch
from ..base_reward_fn import BaseRewardFn

if TYPE_CHECKING:
    from ...navigation import NavigationEnv


class ReachNavigationPointRewardFn(BaseRewardFn):
    def __init__(
        self,
        min_distance_m: float,  # 抵达判定距离阈值
        weight: float = 1,
    ) -> None:
        """单点抵达事件奖励"""
        super().__init__()
        self.min_distance_m = min_distance_m
        self.weight = weight

    def reset(self, env: NavigationEnv, env_indices=None, **kwargs):
        pass

    def forward(self, env: NavigationEnv, plane, **kwargs) -> torch.Tensor:
        distance = env.cur_nav_dist
        reward = torch.zeros_like(distance)
        reached = distance <= self.min_distance_m
        if torch.any(reached):
            indices = torch.where(reached)[0]
            env.cur_nav_point_index[indices] += 1
            reward[indices] += 1

            # 进入新阶段
            for reward_fn in env._reward_fns:
                reward_fn.reset(env, indices)
            if 0 in indices:
                # render
                env.render_navigation_points()
        return reward
