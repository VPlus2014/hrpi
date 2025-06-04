from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from abc import ABC, abstractmethod
from collections.abc import Sequence

from ..base_model import BaseFV, _SupportedIndexType

if TYPE_CHECKING:
    from environments.models.aircraft import BaseAircraft


class BaseMissile(BaseFV):
    STATUS_LAUNCHED = BaseFV.STATUS_ALIVE
    STATUS_HIT = 1
    STATUS_MISSED = 2

    def __init__(
        self,
        target: "BaseAircraft",
        acmi_type="Weapon+Missile",
        **kwargs,  # other parameters
    ) -> None:
        """导弹基类

        Args:
            target (BaseAircraft): 锁定目标
            **kwargs: 其他参数, 参见 BaseFV.__init__
        """
        super().__init__(acmi_type=acmi_type, **kwargs)
        device = self.device
        dtype = self.dtype
        _shape = [self.batchsize]

        self.target = target

        # simulation parameters
        self.demage = 100.0
        self.exp_radius = 20.0  # 毁伤半径 unit: m
        self._t_thrust_s = 3.0  # time limitation of engine, unit: s

        # simulation variables
        self.los = torch.zeros(_shape + [3], device=device, dtype=dtype)
        """到锁定目标的 NED 视线向量 shape (B, 3)"""
        self.distance = torch.full(_shape + [1], 2000e3, device=device, dtype=dtype)
        """最新弹目距离 unit: m shape (B, 1)"""
        self.miss_distance = torch.full(
            _shape + [1], 2000e3, device=device, dtype=dtype
        )
        """脱靶量 unit: m shape (B, 1)"""
        self.distance_history = torch.full(
            _shape + [20], 2000e3, device=device, dtype=dtype
        )
        """最近20次弹目距离 unit: m shape (B, L)"""

    @abstractmethod
    def reset(self, env_indices: torch.Tensor):
        # todo in subclass: 初始化物理运动状态

        self.miss_distance[env_indices] = 400e3
        self.distance_history[env_indices, :] = 400e3

        super().reset(env_indices)

    @abstractmethod
    def launch(self, env_indices: _SupportedIndexType = None):
        env_indices = self.proc_batch_index(env_indices)
        self.status[env_indices] = BaseMissile.STATUS_LAUNCHED

    def is_launch(self, env_indices: _SupportedIndexType = None) -> torch.Tensor:
        env_indices = self.proc_batch_index(env_indices)
        return self.status[env_indices] == BaseMissile.STATUS_LAUNCHED

    def update_distance(self):
        """测量距离&更新脱靶量"""
        los = self.target.position_e() - self.position_e()
        self.los[...] = los

        d = torch.norm(los, p=2, dim=-1)
        self.distance[...] = d

        self.miss_distance = torch.min(self.miss_distance, d)

    def try_hit(self):
        """更新命中状态"""
        d = self.distance
        hit = d <= self.exp_radius
        if torch.any(hit):
            indices = torch.where(hit)[0]

            self.status[indices] = BaseMissile.STATUS_HIT
            self.target.health_point[indices] -= self.demage

    def is_hit(self, env_indices: _SupportedIndexType = None) -> torch.Tensor:
        env_indices = self.proc_batch_index(env_indices)
        return self.status[env_indices] == BaseMissile.STATUS_HIT

    def try_miss(self):
        flag_1 = self.sim_time_s > self._t_thrust_s

        d = self.distance
        self.distance_history = torch.roll(self.distance_history, shifts=-1, dims=-1)
        self.distance_history[..., -1] = d

        diffs = self.distance_history.diff(dim=-1)
        flag_2 = (diffs > 0).all(dim=1)
        flag = flag_1 & flag_2

        if torch.any(flag):
            indices = torch.where(flag)[0]
            self.status[indices] = BaseMissile.STATUS_MISSED
            self.miss_distance[indices] = d[indices]

    def is_missed(self, env_indices: _SupportedIndexType = None) -> torch.Tensor:
        env_indices = self.proc_batch_index(env_indices)
        return self.status[env_indices] == BaseMissile.STATUS_MISSED
