from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from .proto4tc import BaseTerminationFn

if TYPE_CHECKING:
    from ._rt_head import SyncVecEnv, EnvMaskType
    from ..nav_heading import NavHeadingEnv


class TC_ReachGoal(BaseTerminationFn):

    def __init__(self, env: SyncVecEnv, distance_thresh: float, **kwargs) -> None:
        super().__init__(env, **kwargs)
        self._dij_tol = distance_thresh

    def reset(
        self, env: SyncVecEnv, env_mask: EnvMaskType | None = None, **kwargs
    ) -> None:
        pass

    def forward(self, env: NavHeadingEnv, unit, **kwargs) -> np.ndarray:
        yes = env.goal_distance < self._dij_tol
        # if np.any(_terminated):
        #     pass
        return yes
