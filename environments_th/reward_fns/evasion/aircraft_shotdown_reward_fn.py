import torch
from typing import TYPE_CHECKING
from ..base_reward_fn import BaseRewardFn
from environments_th.utils.math_pt import euler_from_quat, ned2aer

if TYPE_CHECKING:
    from environments_th.evasion import EvasionEnv


class AircraftShotdownRewardFn(BaseRewardFn):
    """击落事件惩罚"""

    def __init__(self, weight: float):
        super().__init__()
        self.weight = weight

    def reset(self, env, env_indices, **kwargs):
        pass

    def forward(self, env, plane, **kwargs) -> torch.Tensor:
        return -unit.is_shotdown()
