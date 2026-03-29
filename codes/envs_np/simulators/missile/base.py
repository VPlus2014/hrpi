from __future__ import annotations
from typing import TYPE_CHECKING

# import torch
from abc import abstractmethod

if TYPE_CHECKING:
    # from ..aircraft import BaseAircraft
    pass

from ..proto4model import BaseModelGroup, SupportedMaskType, ACMI_Types
from ._utils import mext as mext


class BaseMissile(BaseModelGroup):
    STATUS_LAUNCHED = BaseModelGroup.STATUS_ALIVE

    RESULT_NONE = 0
    """[命中结果]待定"""
    RESULT_HIT = 1
    """[命中结果]命中"""
    RESULT_MISSED = 2
    """[命中结果]脱靶"""

    def __init__(
        self,
        kill_radius: float = 20.0,  # 杀伤半径 unit: m
        acmi_type=ACMI_Types.Missile.value,
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
        _grp_shape = self.group_shape

        _0 = self._0F

        # simulation parameters
        self.demage = 100.0
        assert kill_radius > 0, "kill_radius must be positive"
        self.kill_radius = mext.empty_like(_0)
        """毁伤半径 unit: m; shape: (...,N,1)"""
        self.kill_radius[...] = kill_radius
        self._t_thrust_s = 3.0  # time limitation of engine, unit: s

        # simulation variables
        self.target_id = mext.empty(
            _grp_shape + (1,),
            # device=device,
            dtype=mext.int64,
        )
        """目标ID, shape: (...,N,1)"""
        self.target_pos_e = mext.empty(
            _grp_shape + (3,),
            # device=device,
            dtype=dtype,
        )
        """目标NED位置, shape: (...,N,3)"""
        self.target_vel_e = mext.empty(
            _grp_shape + (3,),
            # device=device,
            dtype=dtype,
        )
        """目标NED速度, shape: (...,N,3)"""
        #
        self.distance = mext.full(
            _grp_shape + (1,),
            2000e3,
            # device=device,
            dtype=dtype,
        )
        """最新弹目距离 unit: m shape=(...,N,1)"""
        self.miss_distance = mext.full(
            _grp_shape + (1,),
            2000e3,
            # device=device,
            dtype=dtype,
        )
        """脱靶量 unit: m shape=(...,N,1)"""
        self.distance_history = mext.full(
            _grp_shape + (10,),
            2000e3,
            # device=device,
            dtype=dtype,
        )
        """最近若干次弹目距离 unit: m; shape=(...,N,T)"""

        self._result = mext.zeros(
            _grp_shape + (1,),
            # device=device,
            dtype=self.status.dtype,
        )
        """导弹命中状态"""

    @abstractmethod
    def reset(self, mask: SupportedMaskType | None):
        """
        状态复位
        todo in subclass: 初始化物理运动状态
        """
        mask = self.proc_to_mask(mask)
        self.set_result(self.RESULT_NONE, mask)
        d0 = 8000e3
        self.miss_distance[mask, :] = d0
        self.distance_history[mask, :] = d0
        super().reset(mask)

    @abstractmethod
    def launch(self, mask: SupportedMaskType | None):
        """发射"""
        mask = self.proc_to_mask(mask)
        self.set_status(self.STATUS_LAUNCHED, mask)

    def is_launched(self):
        return self.is_alive()

    def set_result(
        self, result: int | mext.ndarray, dst_mask: SupportedMaskType | None
    ):
        """设置命中结果"""
        dst_mask = self.proc_to_mask(dst_mask)
        self._result[dst_mask, :] = result

    def _result_is(self, value: int | mext.ndarray) -> mext.BoolNDArr:
        """命中结果是否为 value"""
        return mext.equal(self._result, value)

    def set_target_info(self, pos_e: mext.ndarray, vel_e: mext.ndarray):
        """
        设置初始目标信息

        Args:
            pos_e (ndarray): _description_
            vel_e (ndarray): _description_
        """
        self.target_pos_e[...] = pos_e
        self.target_vel_e[...] = vel_e

    @abstractmethod
    def observe(self, pos_e: mext.ndarray, vel_e: mext.ndarray, mask: mext.BoolNDArr):
        """
        输入潜在目标信息
        Args:
            unit_pos (ndarray): 物体位置(NED地轴坐标), shape: (N,3)
            unit_vel (ndarray): 物体速度(NED地轴坐标), shape: (N,3)
            mask (ndarray): 掩码, shape: (N,self.batch_size), 物体 i 信息对导弹 j 是否可用
        """
        raise NotImplementedError

    def _update_distance(
        self, new_distance: mext.ndarray, dst_mask: SupportedMaskType | None
    ):
        """测量距离&更新脱靶量
        Args:
            new_distance (ndarray): 最新测量距离, shape: (N,1)
        """
        dst_mask = self.proc_to_mask(dst_mask)
        self.distance[dst_mask, :] = new_distance

        self.miss_distance[dst_mask, :] = mext.minimum(
            self.miss_distance[dst_mask, :], new_distance
        )
        if self.DEBUG:
            self.logger.debug(
                (
                    "distance:{:.3g}".format(self.distance.ravel()[[0]].item()),
                    "MD:{:.3g}".format(self.miss_distance.ravel()[[0]].item()),
                )
            )

    def try_hit(self):
        """更新命中状态"""
        d = self.distance
        hit = self.is_alive() & self.is_no_result() & (d <= self.kill_radius)
        self.set_result(self.RESULT_HIT, hit)
        # if bkbn.any(hit):

    def is_hit(self) -> mext.ndarray:
        """是否命中, shape: (...,N,1)"""
        return self._result_is(self.RESULT_HIT)

    def is_no_result(self) -> mext.BoolNDArr:
        """命中结果待定, shape: (...,N,1)"""
        return self._result_is(self.RESULT_NONE)

    def try_miss(self):
        flag_0 = self.is_alive() & self.is_no_result()
        flag_1 = self.sim_time_s() > self._t_thrust_s  # 滑翔段

        d = self.distance  # (N,1)
        self.distance_history = mext.roll(self.distance_history, shift=-1, axis=-1)
        self.distance_history[..., -1:] = d

        # incs = self.distance_history.diff(dim=-1)
        is_closing = d < self.distance_history[..., 0:1]  # 严格接近
        miss = flag_0 & flag_1 & ~is_closing
        self.set_result(BaseMissile.RESULT_MISSED, miss)

    def is_missed(self) -> mext.ndarray:
        """是否脱靶, shape: (...,N,1)"""
        return self._result_is(self.RESULT_MISSED)
