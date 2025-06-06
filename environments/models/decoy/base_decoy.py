from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from abc import abstractmethod

from ..base_model import BaseModel, BaseModel

if TYPE_CHECKING:
    from ..base_model import _SupportedIndexType
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
            acmi_type (str, optional): Tacview model type
            **kwargs: 其他参数, 参见 BaseFV.__init__
        """
        super().__init__(
            acmi_type=acmi_type, use_eb=use_eb, use_ew=use_ew, use_wb=use_wb, **kwargs
        )
        # device = self.device
        # dtype = self.dtype
        # nenvs = self.batchsize

    @abstractmethod
    def reset(self, env_indices: _SupportedIndexType = None):
        env_indices = self.proc_batch_index(env_indices)

        super().reset(env_indices)

        self.health_point[env_indices] = 100.0

    @abstractmethod
    def run(self):
        super().run()

    def activate(self, env_indices: _SupportedIndexType = None):
        env_indices = self.proc_batch_index(env_indices)
        self.status[env_indices, :] = self.__class__.STATUS_ALIVE
