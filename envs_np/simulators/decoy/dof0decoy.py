from __future__ import annotations
from typing import TYPE_CHECKING, Sequence

# import torch
from abc import abstractmethod
from copy import deepcopy
from .base_decoy import BaseDecoy
from ...utils.math_np import (
    ode_rk23,
    quat_rotate,
    quat_rotate_inv,
    rpy2quat,
    rpy2quat_inv,
    quat_mul,
    ned2aer,
    where,
    ndarray,
    BoolNDArr,
    split_,
    cat,
)

if TYPE_CHECKING:
    from ..base_model import SupportedMaskType as SupportedMaskType
from ...utils.tacview import ACMI_Types

_DEBUG = True


class DOF0BallDecoy(BaseDecoy):

    def __init__(
        self,
        acmi_type=ACMI_Types.Decoy.value,
        vis_radius=100.0,
        effect_duration=10.0,
        use_gravity=True,
        **kwargs,
    ) -> None:
        """
        球形遮蔽 自由落体诱饵

        Args:
            acmi_type (str, optional): Tacview model type
            vis_radius (float, optional): 遮蔽半径, unit: m
            effect_duration (float, optional): 遮蔽持续时间, unit: s
            use_gravity (bool, optional): 是否考虑重力(默认True)
            **kwargs: 其他参数, 参见 BaseFV.__init__
        """
        super().__init__(
            acmi_type=acmi_type,
            use_eb=False,
            use_ew=False,
            use_wb=False,
            use_gravity=use_gravity,
            vis_radius=vis_radius,
            **kwargs,
        )
        # device = self.device
        # dtype = self.dtype
        # nenvs = self.batchsize
        self._effect_duration = effect_duration

    def reset(self, mask: SupportedMaskType | None):
        mask = self.proc_to_mask(mask)
        super().reset(mask)
        self._propagate(mask)

    def run(self, mask: SupportedMaskType | None = None):
        mask = self.proc_to_mask(mask)
        pe, ve = self._run_ode(
            self.sim_step_size_s,
            self.sim_time_s(),
            pos_e=self.position_e(),
            vel_e=self.velocity_e(),
            mask=mask,
        )
        super().run(mask)  # time++
        # 后处理
        self._pos_e[mask, :] = pe[mask, :]
        self._vel_e[mask, :] = ve[mask, :]
        self._propagate(mask)

    def _propagate(self, mask: SupportedMaskType | None):
        logr = self.logr
        mask = self.proc_to_mask(mask)
        is_alive = self.is_alive()
        is_timeout = self.sim_time_s() >= self._effect_duration

        is_new_timeout = is_alive & is_timeout & mask[..., None]
        self.status[...] = where(
            is_new_timeout,
            self.STATUS_DYING,
            self.status,
        )
        self._ppgt_Ve2tas(mask)
        self._ppgt_z2alt(mask)
        if self.DEBUG:
            logr.debug(
                (
                    "status: {}".format(self.status.ravel()[[0]].item()),
                    "new_timeout: {}".format(is_new_timeout.ravel()[[0]].item()),
                )
            )
        return

    def _run_ode(
        self,
        dt_s: float | ndarray,  # 积分步长(不考虑掩码) unit: sec
        t_s: ndarray,  # 初始时间 unit: sec
        pos_e: ndarray,  # 初始位置地轴系坐标
        vel_e: ndarray,  # 初始速度地轴系坐标
        mask: BoolNDArr,
    ):
        r"""求解运动学关键状态转移, 但不修改本体状态\
        Args:
            ...

        Returns:
            pos_e_next (NDArr): 位置地轴系坐标, shape: (...,3)
            tas_next (NDArr): 真空速, shape: (...,1)
            Qew_next (NDArr): 地轴/风轴四元数, shape: (...,4)
        """
        ode_solver = ode_rk23
        logr = self.logr

        dt_s = dt_s * (self.is_alive() & mask[..., None])

        rst = ode_solver(self._f, t_s, cat((pos_e, vel_e), axis=-1), dt_s)
        pos_e_next, vel_e_next = split_(rst, [3, 3], -1)
        if self.DEBUG:
            logr.debug(
                (
                    "id:{}".format(self.acmi_id.ravel()[[0]].item()),
                    "pos_e:{}".format(pos_e_next.reshape(-1, 3)[0, :].tolist()),
                    "vel_e:{}".format(vel_e_next.reshape(-1, 3)[0, :].tolist()),
                )
            )
        return pos_e_next, vel_e_next

    def _f(self, t: ndarray | float, X: ndarray):
        """动力学"""
        p_e, v_e = split_(X, [3, 3], -1)
        a_e = self._0F
        if self.use_gravity:
            a_e = self.g_e()

        dot_p_e = v_e
        dot_v_e = a_e
        dotX = [dot_p_e, dot_v_e]
        dotX = cat(dotX, axis=-1)
        return dotX
