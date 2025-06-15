# import torch
from typing import TYPE_CHECKING, Sequence
from ..proto4rf import BaseRewardFn
from environments_th.utils.math_pt import euler_from_quat, ned2aer

if TYPE_CHECKING:
    from environments_th.evasion import EvasionEnv


class AircraftSurvivalRewardFn(BaseRewardFn):
    def __init__(self, weight: float):
        super().__init__()
        self.weight = weight

    def reset(self, env, env_indices, **kwargs):
        pass

    def forward(self, env: "EvasionEnv", plane, **kwargs) -> np.ndarray:
        return env.missile.is_missed()
