from __future__ import annotations
from abc import ABC, abstractmethod

import torch


class BatchCostFn(ABC):
    """
    向量化代价函数
    """

    @abstractmethod
    def stage_cost(
        self,
        state: torch.Tensor,
        control: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        向量化单步代价计算
        Args:
            state: 状态 shape=(..., dimX)
            control: 控制量 shape=(..., dimU)
        Returns:
            cost: 单步代价 shape=(...,1)
        """
        raise NotImplementedError

    @abstractmethod
    def terminal_cost(self, state: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        向量化终端代价计算(启发式估计)
        Args:
            state: shape=(..., dimX)
        Returns:
            cost: 终端代价 shape=(..., 1)
        """
        raise NotImplementedError

