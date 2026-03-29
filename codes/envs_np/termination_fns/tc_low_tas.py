from __future__ import annotations

# import torch
from typing import TYPE_CHECKING

import numpy as np

from .proto4tc import BaseTerminationFn

if TYPE_CHECKING:
    from ._rt_head import *


class TC_LowTAS(BaseTerminationFn):
    """空速过低"""

    def __init__(self, min_tas_mps: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self._params.update(min_tas_mps=min_tas_mps)
        self._tas_min = min_tas_mps

    def reset(
        self, env: NPSyncVecEnv, env_mask: EnvMaskType | None = None, **kwargs
    ) -> None:
        pass

    def forward(self, env: NPSyncVecEnv, unit, **kwargs) -> np.ndarray:
        return unit.tas() < self._tas_min
