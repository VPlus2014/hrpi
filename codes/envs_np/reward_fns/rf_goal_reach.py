from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..nav_heading import NavHeadingEnv
    from ._rt_head import *
# import torch
from .proto4rf import BaseRewardFn, RewardType


class RF_GoalReach(BaseRewardFn):
    def __init__(
        self,
        min_distance_m: float | None = None,  # 抵达判定距离阈值
        **kwargs,
    ) -> None:
        """单点抵达事件奖励"""
        super().__init__(**kwargs)
        self._params.update(min_distance_m=min_distance_m)
        self.min_distance_m = min_distance_m

    def reset(
        self,
        env: NavHeadingEnv,
        env_mask: EnvMaskType | None = None,
        **kwargs,
    ):
        pass

    def forward(self, env: NavHeadingEnv, unit, **kwargs) -> RewardType:
        if self.min_distance_m is None:
            yes = env.goal_cur_reached
        else:
            yes = env.goal_distance < self.min_distance_m
        reward = yes * 1.0
        return reward
