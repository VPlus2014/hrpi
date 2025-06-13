from __future__ import annotations
from typing import TYPE_CHECKING

# import torch
import math
from ..utils.math_np import euler_from_quat, delta_rad_reg
from .base_reward_fn import BaseRewardFn


class CourseKeepingRewardFn(BaseRewardFn):
    """航向角一致"""

    def __init__(self, course: float, weight: float = 1, version=2) -> None:
        super().__init__()

        self.course = math.atan2(math.sin(course), math.cos(course))
        self.weight = weight
        self._version = version

    def reset(self, env, env_indices=None, **kwargs):
        pass

    def forward(self, env, plane, **kwargs) -> torch.Tensor:
        chi = plane.chi()
        _version = self._version
        if _version == 1:
            err = torch.abs(math.sin(self.course) - torch.sin(chi)) + torch.abs(
                math.cos(self.course) - torch.cos(chi)
            )
        elif _version == 2:
            chi_d = self.course + torch.zeros_like(chi)
            err = delta_rad_reg(chi_d, chi)  # 角误差
        else:
            raise NotImplementedError
        return -err
