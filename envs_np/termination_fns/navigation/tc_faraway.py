from __future__ import annotations

# import torch
from typing import TYPE_CHECKING

import numpy as np
from ..proto4tc import BaseTerminationFn

if TYPE_CHECKING:
    from .._rt_head import SyncVecEnv, EnvMaskType
    from ...navigation import NavigationEnv
    from ...nav_heading import NavHeadingEnv


class TC_AwayFromGoal(BaseTerminationFn):
    def __init__(self,distance_threshold: float, **kwargs) -> None:
        """距离过远"""
        super().__init__(**kwargs)
        self.distance_threshold = distance_threshold

    def reset(
        self, env: SyncVecEnv, env_mask: EnvMaskType | None = None, **kwargs
    ) -> None:
        pass

    def forward(self, env: NavHeadingEnv, unit, **kwargs) -> np.ndarray:
        return env.goal_is_far_away
