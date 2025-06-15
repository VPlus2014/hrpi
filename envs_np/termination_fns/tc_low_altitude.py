from __future__ import annotations

# import torch
from typing import TYPE_CHECKING

import numpy as np
from .proto4tc import BaseTerminationFn

if TYPE_CHECKING:
    from ._rt_head import *


class TC_LowAltitude(BaseTerminationFn):

    def __init__(self, min_altitude_m: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.min_alt_m = min_altitude_m

    def reset(
        self, env: SyncVecEnv, env_mask: EnvMaskType | None = None, **kwargs
    ) -> None:
        pass

    def forward(self, env, unit, **kwargs) -> np.ndarray:
        return unit.altitude_m() < self.min_alt_m
