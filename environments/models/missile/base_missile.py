from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from abc import ABC, abstractmethod
from collections.abc import Sequence

from ..base_model import BaseFV

if TYPE_CHECKING:
    from environments.models.aircraft import BaseAircraft


class BaseMissile(BaseFV):
    STATUS_LAUNCHED = BaseFV.STATUS_ALIVE
    STATUS_HIT = 1
    STATUS_MISSED = 2

    def __init__(
        self,
        target: "BaseAircraft",
        model_type="Missile",
        **kwargs,  # other parameters
    ) -> None:
        """导弹基类

        Args:
            target (BaseAircraft): 锁定目标
            **kwargs: 其他参数, 参见 BaseFV.__init__
        """
        super().__init__(model_type=model_type, **kwargs)
        device = self.device
        dtype = self.dtype
        num_envs = self.batchsize

        self.target = target

        # simulation parameters
        self.demage = 100.0
        self.exp_radius = 20.0  #
        self._t_thrust_s = 3.0  # time limitation of engine, unit: s

        # simulation variables
        self.miss_distance = torch.zeros((num_envs,), device=device, dtype=dtype)
        self.distance_history = torch.full(
            (num_envs, 20), 1e7, device=device, dtype=dtype
        )

    @property
    @abstractmethod
    def velocity_k(self) -> torch.Tensor: ...

    @abstractmethod
    def reset(self, env_indices: torch.Tensor):
        # todo in subclass: 初始化物理运动状态
        
        self.miss_distance[env_indices] = 0.0
        self.distance_history[env_indices, :] = 1e7

        super().reset(env_indices)


    @abstractmethod
    def launch(self, env_indices: Sequence[int] | torch.Tensor | None = None):
        env_indices = self.proc_indices(env_indices)
        self.status[env_indices] = BaseMissile.STATUS_LAUNCHED

    def is_launch(
        self, env_indices: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        env_indices = self.proc_indices(env_indices)
        return self.status[env_indices] == BaseMissile.STATUS_LAUNCHED

    def step_hit(self):
        # 命中逻辑
        d = torch.norm(self.position_e - self.target.position_e, p=2, dim=-1)
        flag = d < self.exp_radius
        if torch.any(flag):
            indices = torch.where(flag)[0]

            self.status[indices] = BaseMissile.STATUS_HIT
            self.target.health_point[indices] -= self.demage
            self.miss_distance[indices] = d[indices]

    def is_hit(
        self, env_indices: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        env_indices = self.proc_indices(env_indices)
        return self.status[env_indices] == BaseMissile.STATUS_HIT

    def step_miss(self):
        flag_1 = self.sim_time_s > self._t_thrust_s

        d = torch.norm(self.target.position_e - self.position_e, dim=-1)
        self.distance_history = torch.roll(self.distance_history, shifts=-1, dims=-1)
        self.distance_history[..., -1] = d

        diffs = self.distance_history.diff(dim=-1)
        flag_2 = (diffs > 0).all(dim=1)
        flag = flag_1 & flag_2

        if torch.any(flag):
            indices = torch.where(flag)[0]
            self.status[indices] = BaseMissile.STATUS_MISSED
            self.miss_distance[indices] = d[indices]

    def is_missed(
        self, env_indices: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        env_indices = self.proc_indices(env_indices)
        return self.status[env_indices] == BaseMissile.STATUS_MISSED
