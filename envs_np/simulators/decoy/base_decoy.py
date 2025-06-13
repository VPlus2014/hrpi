from __future__ import annotations
from typing import TYPE_CHECKING
# import torch
from abc import abstractmethod

from ..base_model import BaseModel, BaseModel

if TYPE_CHECKING:
    from ..base_model import SupportedMaskType
from ...utils.tacview import ACMI_Types


class BaseDecoy(BaseModel):

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
