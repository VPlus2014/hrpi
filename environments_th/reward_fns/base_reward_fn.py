from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...environments_th.proto4venv import (
        TorchSyncVecEnv as TorchSyncVecEnv,
        _EnvIndexType as _EnvIndexType,
    )

    # from ..simulators.base_model import BaseModel, BaseFV
    from ..simulators.aircraft import BaseAircraft as BaseAircraft

import torch
from abc import ABC, abstractmethod


class BaseRewardFn(ABC):

    weight: float = 1.0
    """奖励权重, 默认为1.0"""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def reset(
        self,
        env: TorchSyncVecEnv,
        env_indices: _EnvIndexType = None,
        **kwargs,
    ) -> None:
        """初始化"""
        pass

    @abstractmethod
    def forward(
        self,
        env: TorchSyncVecEnv,
        plane: BaseAircraft,
        **kwargs,
    ) -> torch.Tensor:
        """加权之前的奖励"""
        pass

    def __call__(
        self,
        env: TorchSyncVecEnv,
        plane: BaseAircraft,
        **kwargs,
    ) -> torch.Tensor:
        try:
            rst = self.weight * self.forward(env, plane, **kwargs)
            if isinstance(rst, torch.Tensor):
                assert rst.shape == (env.num_envs, 1), (
                    f"reward shape should be (num_envs,1), but got",
                    rst.shape,
                    "@",
                    self.__class__.__name__,
                )
        except Exception as e:
            raise type(e)(e, self.__class__.__name__)
        return rst
