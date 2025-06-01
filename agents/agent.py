from __future__ import annotations
from typing import TYPE_CHECKING, Union
import torch
import torch.nn as nn
import numpy as np
from functools import cached_property

if TYPE_CHECKING:
    from gymnasium import spaces
    from torch.utils.tensorboard.writer import SummaryWriter
    from typing import TypeAlias
from abc import ABC, abstractmethod

DeviceLikeType: TypeAlias = Union[str, torch.device, int]


class Agent(nn.Module):
    """智能体协议"""

    def __init__(
        self,
        name: str,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        batch_size: int,
        num_envs: int = 1,
        writer: SummaryWriter | None = None,
        device: DeviceLikeType = "cpu",
        dtype: torch.dtype = torch.float,
    ):
        super().__init__()
        self.name = name
        self.observation_space = observation_space
        self.action_space = action_space
        self.batch_size = batch_size
        self.num_envs = num_envs
        self.writer = writer
        self.device = torch.device(device)
        self.dtype = dtype

    def to(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return super().to(device=self.device, dtype=self.dtype)

    @cached_property
    def observation_low(self) -> torch.Tensor:
        return torch.tensor(
            self.observation_space.low, dtype=self.dtype, device=self.device
        )

    @cached_property
    def observation_high(self) -> torch.Tensor:
        return torch.tensor(
            self.observation_space.high, dtype=self.dtype, device=self.device
        )

    @abstractmethod
    def evaluate(self, state: np.ndarray | torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def choose_action(
        self,
        state: np.ndarray | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    """
    根据状态产生动作，并计算对数似然

    Args:
        state (np.ndarray | torch.Tensor): 状态

    Returns:
        tuple[torch.Tensor, torch.Tensor]: 动作值, 动作对数似然
    """

    def _forward(
        self,
        state_normalized: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    """
    根据状态采样动作(action not None 则跳过), 评估动作对数似然
    Args:
        state_normalized (np.ndarray | torch.Tensor): 归一化状态
        action (np.ndarray | torch.Tensor | None, optional): 动作. Defaults to None.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: 动作值, 动作对数似然
    """

    @abstractmethod
    def update(self, global_step: int) -> dict[str, float]: ...
