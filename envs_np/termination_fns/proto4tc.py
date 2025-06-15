from __future__ import annotations
from copy import deepcopy
from functools import cached_property
from typing import TYPE_CHECKING, Any

# import torch
from abc import ABC, abstractmethod

import numpy as np

if TYPE_CHECKING:
    from ..proto4venv import SyncVecEnv, EnvMaskType
    from ..simulators.proto4model import BaseModel


class BaseTerminationFn(ABC):
    def __init__(self, env: SyncVecEnv | None = None, **kwargs) -> None:
        super().__init__()
        self.env = env
        self._params: dict[str, Any] = {**deepcopy(kwargs)}
        """构造参数组"""

    @abstractmethod
    def reset(
        self, env: SyncVecEnv, env_mask: EnvMaskType | None = None, **kwargs
    ) -> None:
        pass

    @abstractmethod
    def forward(self, env: SyncVecEnv, unit: BaseModel, **kwargs) -> np.ndarray:
        pass

    def __call__(self, env: SyncVecEnv, unit: BaseModel, **kwargs) -> np.ndarray:
        try:
            rst = self.forward(env, unit, **kwargs)
            if isinstance(rst, np.ndarray):
                shape = rst.shape
                assert shape == (env.num_envs, 1), (
                    f"expected shape=({env.num_envs},1) got",
                    shape,
                )
        except Exception as e:
            raise type(e)(e, self.__class__.__name__)
        return rst

    def __repr__(self) -> str:
        return "{}({})".format(
            self.__class__.__name__,
            ",".join(f"{k}={repr(v)}" for k, v in self._params.items()),
        )

    @cached_property
    def repr(self) -> str:
        return repr(self)
