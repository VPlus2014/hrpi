from __future__ import annotations
from typing import TYPE_CHECKING

# import torch
from ..proto4model import BaseModelGroup, BaseModelGroup

if TYPE_CHECKING:
    from ..missile import BaseMissile
    from ..decoy import BaseDecoy
    from ..proto4model import SupportedMaskType

from ._utils import mext as mext
from ..proto4model import ACMI_Types


class BasePlane(BaseModelGroup):

    TERMSTATE_NONE = 0  # 无(正常)
    TERMSTATE_CRASH = 0x1  # 撞地坠毁
    TERMSTATE_SHOTDOWN = 0x2  # 被武器击落
    # crash 与 shotdown 互斥?

    missiles: list[BaseMissile]  # 导弹全列表
    decoys: list[BaseDecoy]  # 诱饵全列表

    def __init__(
        self,
        carried_missiles: BaseMissile | None = None,
        acmi_type=ACMI_Types.Plane.value,
        **kwargs,
    ) -> None:
        """飞机基类 BaseAircraft

        Args:
            carried_missiles (BaseMissile | None, optional): 导弹挂载. Defaults to None.
            acmi_type (str, optional): Tacview model type. Defaults to "Aircraft".
            **kwargs: 其他参数, 参见 BaseFV.__init__
        """
        super().__init__(acmi_type=acmi_type, **kwargs)
        grp_shape = self.group_shape
        #
        self.carried_missiles = carried_missiles
        self.termstate = mext.zeros((*grp_shape, 1), dtype=mext.int32)

        self.missiles = []
        self.decoys = []

    def reset(self, mask: SupportedMaskType | None):
        mask = self.proc_to_mask(mask)
        super().reset(mask)

        self.health_point[mask, :] = 100.0
        self.set_termstate(self.TERMSTATE_NONE, mask)

    # def launch_missile(self, target_aircraft: "BaseAircraft") -> None:
    #     if self.missiles_num > 0:
    #         missile = self.carried_missiles.pop()
    #         missile.launch(carrier_aircraft=self, target_aircraft=target_aircraft)
    #         self.launched_missiles.append(missile)

    def set_termstate(
        self, termstate: int | mext.ndarray, mask: SupportedMaskType | None
    ) -> None:
        mask = self.proc_to_mask(mask)
        self.termstate[mask, :] = termstate

    def is_shotdown(self, update: bool = True) -> mext.BoolNDArr:
        if update:
            # update status
            flag = self.health_point <= 1e-6
            if flag.any():
                self.set_termstate(self.TERMSTATE_SHOTDOWN, flag)
        return self._termstate_is(self.TERMSTATE_SHOTDOWN)

    def is_crashed(self, update: bool = True) -> mext.BoolNDArr:
        return self._termstate_is(self.TERMSTATE_CRASH)

    def _termstate_is(self, value: int | mext.ndarray) -> mext.BoolNDArr:
        return mext.equal(self.termstate, value)
