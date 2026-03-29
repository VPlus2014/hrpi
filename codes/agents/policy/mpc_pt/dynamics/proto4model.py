from __future__ import annotations
from abc import ABC, abstractmethod
import torch


class BatchDynamics(ABC):
    """
    支持向量化的动力学模型基类
    """

    dimX: int
    dimU: int

    def __init__(self, dimX: int, dimU: int, dt: float = 0.1):
        super().__init__()
        self.dimX = dimX
        self.dimU = dimU
        self.dt = dt

    @abstractmethod
    def forward(
        self, state: torch.Tensor, control: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """
        离散时间动力学
        Args:
            state: 当前状态, shape=(..., dimX)
            control: 控制输入, shape=(..., dimU)
        Returns:
            next_state: 后继状态, shape=(..., dimX)
        """
        raise NotImplementedError("Subclass must implement forward method")

    def __call__(self, state: torch.Tensor, control: torch.Tensor, *args, **kwargs):
        """see `forward`"""
        return self.forward(state, control, *args, **kwargs)

