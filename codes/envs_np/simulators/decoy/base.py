from __future__ import annotations
from typing import TYPE_CHECKING

# import torch
from abc import abstractmethod

from ..proto4model import BaseModelGroup, BaseModelGroup

if TYPE_CHECKING:
    from ..proto4model import SupportedMaskType

from ._utils import *


class BaseDecoy(BaseModelGroup):

    def __init__(
        self,
        acmi_type=ACMI_Types.Decoy.value,
        use_eb=False,
        use_ew=False,
        use_wb=False,
        **kwargs,
    ) -> None:
        """诱饵基类 BaseDecoy

        Args:
            参见 BaseModel.__init__
        """
        super().__init__(
            acmi_type=acmi_type, use_eb=use_eb, use_ew=use_ew, use_wb=use_wb, **kwargs
        )
        # device = self.device
        # dtype = self.dtype
        # nenvs = self.batchsize

    @abstractmethod
    def reset(self, mask: SupportedMaskType | None):
        super().reset(mask)

    @abstractmethod
    def run(self, mask: SupportedMaskType | None = None):
        return super().run(mask)
