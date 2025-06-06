from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from abc import abstractmethod
from collections.abc import Sequence
from copy import deepcopy

from ..base_model import BaseModel, BaseModel
from ...utils.math_torch import (
    quat_rotate,
    quat_rotate_inv,
    rpy2quat,
    rpy2quat_inv,
    quat_mul,
    ned2aer,
)

if TYPE_CHECKING:
    from ..missile import BaseMissile
    from ..base_model import _SupportedIndexType


class BaseAircraft(BaseModel):
    STATUS_CRASH = 1  # 坠毁
    STATUS_SHOTDOWN = 2  # 被武器击落
    # todo: crash 与 shotdown 互斥?

    def __init__(
        self,
        carried_missiles: BaseMissile | None = None,
        acmi_type="Plane",
        **kwargs,
    ) -> None:
        """飞机基类 BaseAircraft

        Args:
            carried_missiles (BaseMissile | None, optional): 导弹挂载. Defaults to None.
            acmi_type (str, optional): Tacview model type. Defaults to "Aircraft".
            **kwargs: 其他参数, 参见 BaseFV.__init__
        """
        super().__init__(acmi_type=acmi_type, **kwargs)
        device = self.device
        dtype = self.dtype
        nenvs = self.batch_size

        #
        self.carried_missiles = carried_missiles

    def reset(self, env_indices: _SupportedIndexType = None):
        env_indices = self.proc_batch_index(env_indices)

        super().reset(env_indices)

        self.health_point[env_indices] = 100.0

    def activate(self, env_indices: _SupportedIndexType = None):
        env_indices = self.proc_batch_index(env_indices)
        self.status[env_indices, :] = BaseAircraft.STATUS_ALIVE

    # def launch_missile(self, target_aircraft: "BaseAircraft") -> None:
    #     if self.missiles_num > 0:
    #         missile = self.carried_missiles.pop()
    #         missile.launch(carrier_aircraft=self, target_aircraft=target_aircraft)
    #         self.launched_missiles.append(missile)

    def crash(self, env_indices: _SupportedIndexType = None):
        env_indices = self.proc_batch_index(env_indices)
        self.status[env_indices] = BaseAircraft.STATUS_CRASH

    def is_crash(self, env_indices: _SupportedIndexType = None) -> torch.Tensor:
        env_indices = self.proc_batch_index(env_indices)
        return self.status[env_indices] == BaseAircraft.STATUS_CRASH

    def is_shotdown(
        self, env_indices: _SupportedIndexType = None, update=True
    ) -> torch.Tensor:

        if update:
            # update status
            flag = self.health_point <= 1e-6
            if torch.any(flag):
                indices = torch.where(flag)[0]

                self.status[indices] = BaseAircraft.STATUS_SHOTDOWN

        env_indices = self.proc_batch_index(env_indices)
        return self.status[env_indices] == BaseAircraft.STATUS_SHOTDOWN
