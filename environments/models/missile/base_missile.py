import torch
from abc import ABC, abstractmethod
from typing import Literal, TYPE_CHECKING
from collections.abc import Sequence

from ..base_model import BaseModel

if TYPE_CHECKING:
    from environments.models.aircraft import BaseAircraft


class BaseMissile(BaseModel):
    STATUS_LAUNCHED = 0
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
            **kwargs: 其他参数, 参见 BaseModel.__init__
        """
        super().__init__(model_type=model_type, **kwargs)
        device = self.device
        dtype = self.dtype
        num_envs = self.batchsize

        self.target = target

        # simulation parameters
        self.demage = 100.0
        self.exp_radius = 20.0  #
        self._t_thrust_s = 3  # time limitation of engine, unit: s

        # simulation variables
        self._tas = torch.zeros(
            (num_envs, 1), device=device, dtype=dtype
        )  # true air speed, unit: m/s, shape: [num_models, 1]
        self.miss_distance = torch.zeros((num_envs,), device=device, dtype=dtype)
        self.distance_history = torch.full(
            (num_envs, 20), 1e7, device=device, dtype=dtype
        )
        self._vel_a = torch.zeros(
            (num_envs, 3), device=device, dtype=dtype
        )  # NED 速度系

    @property
    def tas(self) -> torch.Tensor:
        """true air speed, unit: m/s, shape: [num_models, ]"""
        return self._tas

    @tas.setter
    def tas(self, value: torch.Tensor):
        self._tas.copy_(value)
        self._ppgt_tas2va()

    @property
    def velocity_a(self) -> torch.Tensor:
        return self._vel_a

    @property
    @abstractmethod
    def velocity_k(self) -> torch.Tensor: ...

    def reset(self, env_indices: torch.Tensor):
        self._tas[env_indices] = 0.0
        self._ppgt_tas2va()

        self.miss_distance[env_indices] = 0.0
        self.distance_history[env_indices, :] = 1e7

        super().reset(env_indices)

    def launch(self, env_indices: Sequence[int] | torch.Tensor | None = None):
        env_indices = self.proc_indices(env_indices)
        self._status[env_indices] = BaseMissile.STATUS_LAUNCHED
        self._tas[env_indices] = 600.0
        self._ppgt_tas2va()

    def is_launch(
        self, env_indices: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        env_indices = self.proc_indices(env_indices)
        return self._status[env_indices] == BaseMissile.STATUS_LAUNCHED

    def step_hit(self):
        # 命中逻辑
        d = torch.norm(self.position_g - self.target.position_g, p=2, dim=-1)
        flag = d < self.exp_radius
        if torch.any(flag):
            indices = torch.where(flag)[0]

            self._status[indices] = BaseMissile.STATUS_HIT
            self.target.health_point[indices] -= self.demage
            self.miss_distance[indices] = d[indices]

    def is_hit(
        self, env_indices: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        env_indices = self.proc_indices(env_indices)
        return self._status[env_indices] == BaseMissile.STATUS_HIT

    def step_miss(self):
        flag_1 = 0.001 * self.sim_time_ms > self._t_thrust_s

        d = torch.norm(self.target.position_g - self.position_g, p=2, dim=-1)
        self.distance_history = torch.roll(self.distance_history, shifts=-1, dims=1)
        self.distance_history[..., -1] = d

        diffs = self.distance_history.diff(dim=-1)
        flag_2 = (diffs > 0).all(dim=1)
        flag = flag_1 & flag_2

        if torch.any(flag):
            indices = torch.where(flag)[0]
            self._status[indices] = BaseMissile.STATUS_MISSED
            self.miss_distance[indices] = d[indices]

    def is_missed(
        self, env_indices: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        env_indices = self.proc_indices(env_indices)
        return self._status[env_indices] == BaseMissile.STATUS_MISSED

    def _ppgt_tas2va(self):
        """将真空速转换为速度系坐标"""
        self._vel_a[..., 0:1] = self._tas
