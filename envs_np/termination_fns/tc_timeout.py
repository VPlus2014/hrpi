from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from .proto4tc import BaseTerminationFn

if TYPE_CHECKING:
    from ._rt_head import *


class TC_Timeout(BaseTerminationFn):
    """超时"""

    def __init__(self, timeout_s: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self._timeout_s = timeout_s if timeout_s > 0 else np.inf

    def reset(
        self, env: SyncVecEnv, env_mask: EnvMaskType | None = None, **kwargs
    ) -> None:
        pass

    def forward(self, env, unit, **kwargs) -> np.ndarray:
        trunc = env.sim_time_s >= self._timeout_s
        # if np.any(_terminated):
        #     pass
        return trunc
