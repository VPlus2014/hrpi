from __future__ import annotations

# import torch
from typing import TYPE_CHECKING

import numpy as np
from ..proto4tc import BaseTerminationFn

if TYPE_CHECKING:
    from ...nav_heading import NavHeadingEnv
    from .._rt_head import *


class TC_ReachAllGoal(BaseTerminationFn):
    """抵达了全部目标点"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def reset(
        self, env: SyncVecEnv, env_mask: EnvMaskType | None = None, **kwargs
    ) -> None:
        pass

    def forward(self, env: NavHeadingEnv, unit, **kwargs) -> np.ndarray:
        yes = env.goal_all_reached
        return yes
        # (B,1)
