from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from ..proto4venv import TrueSyncVecEnv
    from ..models.aircraft import BaseAircraft


class BaseTerminationFn(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def reset(self, env: TrueSyncVecEnv, **kwargs) -> None:
        pass

    @abstractmethod
    def forward(self, env: TrueSyncVecEnv, plane: BaseAircraft, **kwargs) -> torch.Tensor:
        pass

    def __call__(self, env: TrueSyncVecEnv, plane: BaseAircraft, **kwargs) -> torch.Tensor:
        rst = self.forward(env, plane, **kwargs)
        shape = rst.shape
        assert shape == (env.num_envs, 1), (
            "expected shape=(-1,1) got",
            shape,
            "@",
            self.__class__.__name__,
        )
        return rst
