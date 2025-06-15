from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..nav_heading import NavHeadingEnv
    from ._rt_head import *
# import torch
from .proto4rf import BaseRewardFn, RewardType


class RF_GoalFar(BaseRewardFn):
    def __init__(
        self,
        max_distance_m: float,  # 抵达判定距离阈值
        **kwargs,
    ) -> None:
        """距离过远事件奖励"""
        super().__init__(**kwargs)
        self._params.update({"min_distance_m": max_distance_m})
        self.min_distance_m = max_distance_m

    def reset(
        self,
        env: NavHeadingEnv,
        env_mask: EnvMaskType | None = None,
        **kwargs,
    ):
        pass

    def forward(self, env: NavHeadingEnv, unit, **kwargs) -> RewardType:
        # distance = env.goal_distance
        # reached = distance <= self.min_distance_m
        reward = -1.0 * env.goal_is_far_away
        return reward
