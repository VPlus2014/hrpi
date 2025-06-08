from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from abc import abstractmethod

if TYPE_CHECKING:
    from ..aircraft import BaseAircraft

from ..base_model import BaseModel, BaseModel, _SupportedIndexType


class BaseMissile(BaseModel):
    STATUS_LAUNCHED = BaseModel.STATUS_ALIVE

    RESULT_NONE = 0
    """[命中结果]待定"""
    RESULT_HIT = 1
    """[命中结果]命中"""
    RESULT_MISSED = 2
    """[命中结果]脱靶"""

    def __init__(
        self,
        kill_radius: float = 20.0,  # 杀伤半径 unit: m
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
        _shape = [self.batch_size]

        _0 = torch.zeros(_shape + [1], device=device, dtype=dtype)

        # simulation parameters
        self.demage = 100.0
        assert kill_radius > 0, "kill_radius must be positive"
        self.kill_radius = torch.empty_like(_0)
        """毁伤半径 unit: m; shape: (N,1)"""
        self.kill_radius.copy_(_0 + kill_radius)
        self._t_thrust_s = 3.0  # time limitation of engine, unit: s

        # simulation variables
        self.target_id = torch.empty(_shape + [1], device=device, dtype=torch.int64)
        """目标ID, shape: (N, 1)"""
        self.target_pos_e = torch.empty(_shape + [3], device=device, dtype=dtype)
        """目标NED位置, shape: (N,3)"""
        self.target_vel_e = torch.empty(_shape + [3], device=device, dtype=dtype)
        """目标NED速度, shape: (N,3)"""
        #
        self.distance = torch.full(_shape + [1], 2000e3, device=device, dtype=dtype)
        """最新弹目距离 unit: m shape (N, 1)"""
        self.miss_distance = torch.full(
            _shape + [1], 2000e3, device=device, dtype=dtype
        )
        """脱靶量 unit: m shape (N, 1)"""
        self.distance_history = torch.full(
            _shape + [10], 2000e3, device=device, dtype=dtype
        )
        """最近若干次弹目距离 unit: m; shape (N, T)"""

        self._result = torch.zeros(_shape + [1], device=device, dtype=torch.int64)
        """导弹命中状态"""

    @abstractmethod
    def reset(self, env_indices: _SupportedIndexType | None):
        """状态复位"""
        self._result[env_indices] = BaseMissile.RESULT_NONE
        self.miss_distance[env_indices] = 400e3
        self.distance_history[env_indices, :] = 400e3
        # todo in subclass: 初始化物理运动状态

        super().reset(env_indices)

    @abstractmethod
    def launch(self, env_indices: _SupportedIndexType | None):
        """发射"""
        env_indices = self.proc_index(env_indices)
        self.set_status(self.STATUS_LAUNCHED, env_indices)

    def is_launch(self) -> torch.Tensor:
        return self.is_alive()

    def set_hit(self):
        self._result[...] = BaseMissile.RESULT_HIT

    def set_target_info(self, pos_e: torch.Tensor, vel_e: torch.Tensor):
        """
        设置初始目标信息

        Args:
            pos_e (torch.Tensor): _description_
            vel_e (torch.Tensor): _description_
        """
        self.target_pos_e[...] = pos_e
        self.target_vel_e[...] = vel_e

    @abstractmethod
    def observe(self, pos_e: torch.Tensor, vel_e: torch.Tensor, mask: torch.BoolTensor):
        """
        输入潜在目标信息
        Args:
            unit_pos (torch.Tensor): 物体位置(NED地轴坐标), shape: (N,3)
            unit_vel (torch.Tensor): 物体速度(NED地轴坐标), shape: (N,3)
            mask (torch.Tensor): 掩码, shape: (N,self.batch_size), 物体 i 信息对导弹 j 是否可用
        """
        raise NotImplementedError

    def _update_distance(
        self, new_distance: torch.Tensor, dst_index: _SupportedIndexType | None
    ):
        """测量距离&更新脱靶量
        Args:
            new_distance (torch.Tensor): 最新测量距离, shape: (N,1)
        """
        dst_index = self.proc_index(dst_index)
        self.distance[dst_index] = new_distance
        self.miss_distance[dst_index] = torch.min(
            self.miss_distance[dst_index], new_distance
        )
        if self.DEBUG:
            self.logr.debug(
                (
                    "distance:{:.3g}".format(self.distance[0, 0].item()),
                    "MD:{:.3g}".format(self.miss_distance[0, 0].item()),
                )
            )

    def try_hit(self):
        """更新命中状态"""
        d = self.distance
        hit = (
            self.is_alive()
            & (self._result == self.RESULT_NONE)
            & (d <= self.kill_radius)
        )
        if torch.any(hit):
            indices = torch.where(hit)[0]

            self._result[indices] = BaseMissile.RESULT_HIT
        # return indices

    def is_hit(self, env_indices: _SupportedIndexType | None) -> torch.Tensor:
        """是否命中"""
        env_indices = self.proc_index(env_indices)
        return self._result[env_indices] == BaseMissile.RESULT_HIT

    def try_miss(self):
        flag_0 = self.is_alive() & (self._result == self.RESULT_NONE)
        flag_1 = self.sim_time_s() > self._t_thrust_s

        d = self.distance  # (N,1)
        self.distance_history = torch.roll(self.distance_history, shifts=-1, dims=-1)
        self.distance_history[..., -1:] = d

        incs = self.distance_history.diff(dim=-1)
        flag_2 = (incs > 0).all(dim=1)  # 距离一直增大
        flag = flag_0 & flag_1 & flag_2

        if torch.any(flag):
            indices = torch.where(flag)[0]
            self._result[indices] = BaseMissile.RESULT_MISSED
            # self.miss_distance[indices] = d[indices]

    def is_missed(self, env_indices: _SupportedIndexType | None) -> torch.Tensor:
        env_indices = self.proc_index(env_indices)
        return self._result[env_indices] == BaseMissile.RESULT_MISSED
