import torch
import torch.nn as nn
import numpy as np
from functools import cached_property
from gymnasium import spaces
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod

class Agent(nn.Module):
    def __init__(
        self,
        name: str, 
        observation_space: spaces.Box,
        action_space: spaces.Box,
        batch_size: int,
        num_envs: int = 1,
        writer: SummaryWriter | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.name = name
        self.observation_space = observation_space
        self.action_space = action_space
        self.batch_size = batch_size
        self.num_envs = num_envs
        self.writer = writer
        self.device = device

    @cached_property
    def observation_low(self) -> torch.Tensor:
        return torch.from_numpy(self.observation_space.low).to(device=self.device)
    
    @cached_property
    def observation_high(self) -> torch.Tensor:
        return torch.from_numpy(self.observation_space.high).to(device=self.device)
    
    @abstractmethod
    def evaluate(
        self, 
        state: np.ndarray | torch.Tensor
    ) -> torch.Tensor:
        ...

    @abstractmethod
    def choose_action(
        self,
        state: np.ndarray | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    @abstractmethod
    def update(
        self,
        global_step: int
    ) -> dict[str, float]:
        ...
