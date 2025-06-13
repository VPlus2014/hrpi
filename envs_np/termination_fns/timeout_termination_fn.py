from __future__ import annotations
from typing import TYPE_CHECKING
# import torch
from .base_termination_fn import BaseTerminationFn

if TYPE_CHECKING:
    from ..proto4venv # import torchSyncVecEnv


class TimeoutTerminationFn(BaseTerminationFn):
    def __init__(self, timeout_s: float) -> None:
        super().__init__()
        self.timeout_s = timeout_s

    def reset(self, env, **kwargs) -> None:
        pass

    def forward(self, env, plane, **kwargs) -> torch.Tensor:
        _terminated = env.sim_time_s >= self.timeout_s
        # if torch.any(_terminated):
        #     pass
        return _terminated
