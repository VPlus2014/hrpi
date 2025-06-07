from __future__ import annotations
from typing import TYPE_CHECKING, Sequence
import torch
from abc import abstractmethod
from copy import deepcopy
from .base_decoy import BaseDecoy
from ...utils.math_torch import (
    ode_rk23,
    quat_rotate,
    quat_rotate_inv,
    rpy2quat,
    rpy2quat_inv,
    quat_mul,
    ned2aer,
)

if TYPE_CHECKING:
    from ..base_model import _SupportedIndexType
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

    def reset(self, env_indices: _SupportedIndexType = None):
        env_indices = self.proc_batch_index(env_indices)
        super().reset(env_indices)
        self._propagate(env_indices)

    def run(self):
        pe, ve = self._run_ode(
            self.sim_step_size_s,
            self.sim_time_s(),
            pos_e=self.position_e(),
            vel_e=self.velocity_e(),
        )
        super().run()  # time++
        # 后处理
        self._pos_e.copy_(pe)
        self._vel_e.copy_(ve)
        self._propagate()

    def _propagate(self, index: _SupportedIndexType = None):
        logr = self.logr
        index = self.proc_batch_index(index)
        is_alive = self.is_alive(index)
        is_timeout = self.sim_time_s(index) >= self._effect_duration

        is_new_timeout = is_alive & is_timeout
        self.status[index] = torch.where(
            is_new_timeout,
            self.STATUS_DYING,
            self.status[index],
        )
        if _DEBUG:
            if is_new_timeout[0].item():
                logr.debug(
                    (
                        "obj: {}".format(self.__class__.__name__),
                        "status: {}".format(self.status[0].item()),
                        "new_timeout: {}".format(is_new_timeout[0].item()),
                    )
                )
        self._ppgt_Ve2tas(index)
        return

    def _run_ode(
        self,
        dt_s: float | torch.Tensor,  # 积分步长(不考虑掩码) unit: sec
        t_s: torch.Tensor,  # 初始时间 unit: sec
        pos_e: torch.Tensor,  # 初始位置地轴系坐标
        vel_e: torch.Tensor,  # 初始速度地轴系坐标
    ):
        r"""求解运动学关键状态转移, 但不修改本体状态\
        Args:
            ...

        Returns:
            pos_e_next (torch.Tensor): 位置地轴系坐标, shape: (B,3)
            tas_next (torch.Tensor): 真空速, shape: (B,1)
            Qew_next (torch.Tensor): 地轴/风轴四元数, shape: (B,4)
        """
        use_gravity = self.use_gravity  # 考虑重力
        ode_solver = ode_rk23
        logr = self.logr

        _0f = self._0f

        def _f(
            t: torch.Tensor | float,
            X: Sequence[torch.Tensor],
        ):
            """动力学"""
            p_e, v_e = X
            a_e = _0f
            if use_gravity:
                a_e = self.g_e()

            dot_p_e = v_e
            dot_v_e = a_e
            dotX = [dot_p_e, dot_v_e]
            return dotX

        dt_s = dt_s * self.is_alive()
        pos_e_next, vel_e_next = ode_solver(_f, (pos_e, vel_e), t_s, dt_s)
        if _DEBUG:
            logr.debug(
                {
                    "obj": self.__class__.__name__,
                    "pos_e": pos_e_next[0].cpu().tolist(),
                    "vel_e": vel_e_next[0].cpu().tolist(),
                    "tas": vel_e_next[0].norm().item(),
                }
            )
        return pos_e_next, vel_e_next
