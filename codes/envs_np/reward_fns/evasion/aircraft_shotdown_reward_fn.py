# import torch
from typing import TYPE_CHECKING
from ..proto4rf import BaseRewardFn
from codes.utils.math_pt import euler_from_quat, ned2aer

if TYPE_CHECKING:
    from envs_pt.evasion import EvasionEnv


class AircraftShotdownRewardFn(BaseRewardFn):
    """击落事件惩罚"""

    def __init__(self, weight: float):
        super().__init__()
        self.weight = weight

    def reset(self, env, env_indices, **kwargs):
        pass

    def forward(self, env, plane, **kwargs) -> np.ndarray:
        return -unit.is_shotdown()
