from __future__ import annotations
from copy import deepcopy
from functools import cached_property
from typing import TYPE_CHECKING, Any, Union, SupportsFloat

if TYPE_CHECKING:
    from ._rt_head import *

from abc import ABC, abstractmethod
from numpy import ndarray, number, floating

RewardType = Union[ndarray, number, float, int, bool]


class BaseRewardFn(ABC):
    """奖励权重, 默认为1.0"""

    def __init__(
        self,
        env: SyncVecEnv | None = None,
        weight: float | SupportsFloat = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.env = env
        self.weight = float(weight)
        self._params: dict[str, Any] = {"weight": weight, **deepcopy(kwargs)}
        """构造参数组"""

    @abstractmethod
    def reset(
        self,
        env: SyncVecEnv,
        env_mask: EnvMaskType | None = None,
        **kwargs,
    ) -> None:
        """奖励函数复位"""
        pass

    @abstractmethod
    def forward(
        self,
        env: SyncVecEnv,
        unit: BaseModel,
        **kwargs,
    ) -> RewardType:
        """加权之前的奖励"""
        pass

    def __call__(
        self,
        env: SyncVecEnv,
        unit: BaseModel,
        **kwargs,
    ) -> RewardType:
        try:
            rst = self.weight * self.forward(env, unit, **kwargs)
            if isinstance(rst, ndarray):
                assert rst.shape == (env.num_envs, 1), (
                    f"reward array shape should be (num_envs,1)=({env.num_envs},1), but got",
                    rst.shape,
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
