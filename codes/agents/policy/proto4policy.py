from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any, Literal, Self, Union
import torch
import torch.nn as nn
import numpy as np
from functools import cached_property

if TYPE_CHECKING:
    from gymnasium import spaces
    from torch.utils.tensorboard.writer import SummaryWriter
    from typing_extensions import TypeAlias
    from ..replay_buffer.proto4data import RolloutBatchProtocol



from tianshou.policy.base import TLearningRateScheduler, TTrainingStats
from ._types import *
from tianshou.data.batch import Batch, BatchProtocol
from ..replay_buffer.proto4data import BaseReplayBuffer
from .utils.net import NetBase, TRecurrentState, NNModule


DeviceLikeType: TypeAlias = Union[str, torch.device, int]
ACTION_BOUND_METHODS = Literal["clip", "tanh"]


class BaseNNPolicy(NNModule, _BasePolicy[TTrainingStats]):

    name: str

    def set_name(self, name: str) -> None:
        self.name = name

    def to(
        self,
        device: DeviceLikeType | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> Self:
        rst = NNModule.to(self, device=device, dtype=dtype, non_blocking=non_blocking)
        assert rst is self
        return self

    def write_stats(self, stat: TTrainingStats, writer: SummaryWriter, step: int):
        pass


