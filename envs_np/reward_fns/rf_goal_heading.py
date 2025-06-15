from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from .proto4rf import BaseRewardFn, RewardType

if TYPE_CHECKING:
    from ..nav_heading import NavHeadingEnv
    from ._rt_head import BaseAircraft, EnvMaskType


class RF_GoalHeadingAngle(BaseRewardFn):
    """
    目标点朝向一致奖励
    """

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

    def reset(
        self,
        env: NavHeadingEnv,
        env_mask: EnvMaskType | None = None,
        **kwargs,
    ):
        from ._rt_head import math_np

    def forward(self, env: NavHeadingEnv, plane, **kwargs) -> RewardType:
        ata = env.goal_ATA
        reward = -ata
        return reward
