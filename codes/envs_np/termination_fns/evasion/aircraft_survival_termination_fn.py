# import torch
from typing import TYPE_CHECKING

import numpy as np
from ..proto4tc import BaseTerminationFn

if TYPE_CHECKING:
    from ...evasion import EvasionEnv


class AircraftSurvivalTerminationFn(BaseTerminationFn):
    def __init__(self) -> None:
        super().__init__()

    def reset(self, env: "EvasionEnv", **kwargs) -> None:
        pass

    def forward(self, env: "EvasionEnv", **kwargs) -> np.ndarray:
        return env.missile.is_missed()
