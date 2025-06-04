from __future__ import annotations
import logging
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

    logr: logging.Logger = logging.getLogger(__name__)

    def __init__(
        self,
        name: str,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        buffer_size: int = 1000000,
        batch_size: int = 128,
        num_envs: int = 1,
        writer: SummaryWriter | None = None,
        device: DeviceLikeType = "cpu",
        dtype: torch.dtype = torch.float,
    ):
        """
        智能体协议

        Args:
            name (str): _description_
            observation_space (spaces.Box): _description_
            action_space (spaces.Box): _description_
            buffer_size (int): 回放池采样容量
            batch_size (int): 训练 batch size
            num_envs (int, optional): 并行写入环境数. Defaults to 1.
            writer (SummaryWriter | None, optional): TensorBoard 记录器. Defaults to None.
            device (DeviceLikeType, optional): 计算设备. Defaults to "cpu".
            dtype (torch.dtype, optional): 数值精度. Defaults to torch.float.
        """
        super().__init__()
        self.name = name
        self.observation_space = observation_space
        self.action_space = action_space
        self.buffer_size = buffer_size
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

    @property
    def observation_min(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def observation_max(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def action_min(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def action_max(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, state: np.ndarray | torch.Tensor) -> torch.Tensor:
        """根据状态产生贪心动作

        Args:
            state (np.ndarray | torch.Tensor): 环境原始状态

        Returns:
            torch.Tensor: 环境动作
        """
        pass

    @abstractmethod
    def choose_action(
        self,
        state: np.ndarray | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """根据状态产生随机动作，并计算对数似然

            state (np.ndarray | torch.Tensor): 环境状态

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 环境动作值, 动作对数似然
        """
        pass

    @abstractmethod
    def update(self, global_step: int) -> dict[str, float]: ...
