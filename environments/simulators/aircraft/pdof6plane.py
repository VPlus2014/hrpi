from __future__ import annotations
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..base_model import _SupportedIndexType
import torch
import numpy as np
from collections.abc import Sequence
from .base_aircraft import BaseModel, BaseAircraft

_DEBUG = True

# from .base_aircraft import BaseMissile
from ...utils.math_pt import (
    normalize,
    quat_rotate,
    quat_mul,
    quat_conj,
    ode_rk45,
    ode_rk23,
    ode_euler,
    delta_rad_reg,
    affcmb,
)

_PI = math.pi


class PDOF6Plane(BaseAircraft):
    def __init__(
        self,
        nx_max: float = 1.0,
        nx_min: float = 0.5,
        ny_max: float = 1.0,
        nz_up_max: float = 8.0,
        nz_down_max: float = 1.0,
        Vmin: float = 240,
        Vmax: float = 240,
        dmu_max: float = math.radians(360 / 4),
        use_gravity: bool = False,
        **kwargs,
    ) -> None:
        """伪DOF6刚体飞机(无转动惯量 & 过载控制 & 只有速度系)

        Args:
            tas (torch.Tensor|float): 初始真空速 unit: m/s, shape: (N, 1), default: 0
            rpy_ew (torch.Tensor|float): 初始速度系姿态欧拉角(mu,gamma,chi) unit: rad, shape: (N, 3), default: 0

            **kwargs: 其他参数, 见 BaseAircraft
        """
        super().__init__(
            use_eb=False,
            use_ew=True,
            use_wb=False,
            use_gravity=use_gravity,
            **kwargs,
        )
        device = self.device
        dtype = self.dtype
        bsz = self.batch_size

        assert ny_max >= 0, ("ny_max must be non-negative.", ny_max)
        assert nx_max >= nx_min, ("nx_max>=nx_min", nx_max, nx_min)
        assert nx_max >= 0, ("nx_max must be non-negative.", nx_max)
        assert nz_up_max >= 0, (
            "nz_up_max must be >0.",
            nz_up_max,
        )
        assert nz_down_max >= 0, (
            "nz_down_max must be >0.",
            nz_down_max,
        )
        assert dmu_max > 0, (
            "dot_mu_max must be positive.",
            dmu_max,
        )
        assert Vmin <= Vmax, (
            "Vmin must be less than or equal to Vmax.",
            Vmin,
            Vmax,
        )
        assert Vmin > 0, "Vmin must be positive."
        # simulation parameters
        self._nx_max = nx_max
        assert nx_min < 0, "nx_min must be negative."
        assert nx_max > 0, "nx_max must be positive."
        self._nx_min = nx_min
        self._nz_up_max = nz_up_max
        self._nz_down_max = nz_down_max
        self._ny_max = ny_max
        self._Vmin = Vmin
        self._Vmax = Vmax
        self._dot_mu_max = dmu_max

        # initial conditions
        self._ic_tas = torch.zeros((bsz, 1), device=device, dtype=dtype)
        self._ic_rpy_ew = torch.zeros((bsz, 3), device=device, dtype=dtype)
        self._ic_pos_e = torch.zeros((bsz, 3), device=device, dtype=dtype)

        # 当前控制量
        self._n_w = torch.zeros((bsz, 3), device=device, dtype=dtype)
        self._dmu = torch.zeros((bsz, 1), device=device, dtype=dtype)

    def set_ic_tas(
        self, tas: torch.Tensor | float, dst_index: _SupportedIndexType | None
    ):
        """设置初始真空速"""
        self._ic_tas[...] = tas

    def set_ic_rpy_ew(
        self, rpy_ew: torch.Tensor | float, dst_index: _SupportedIndexType | None
    ):
        """设置初始速度系姿态欧拉角"""
        self._ic_rpy_ew[...] = rpy_ew

    def set_ic_pos_e(
        self, position_e: torch.Tensor | float, dst_index: _SupportedIndexType | None
    ):
        """设置初始位置地轴系坐标"""
        self._ic_pos_e[...] = position_e

    def reset(
        self,
        env_indices: _SupportedIndexType | None,
    ):
        env_indices = self.proc_index(env_indices)

        super().reset(env_indices)
        self._rpy_ew[env_indices] = self._ic_rpy_ew[env_indices]
        self._tas[env_indices] = self._ic_tas[env_indices]
        self._pos_e[env_indices] = self._ic_pos_e[env_indices]

        self._ppgt_rpy_ew2Qew(env_indices)
        self._propagate(env_indices)
        pass

    def set_action(self, action: np.ndarray | torch.Tensor):
        """
        控制量约定: 全部在 [-1,1] 内!
        Args:
            action (np.ndarray|torch.Tensor): 控制量, shape: (N,4), 分别为:
                - nx_cmd: 期望切向过载指令
                - ny_cmd: 期望横向过载指令
                - nz_cmd: 期望法向过载指令(按照惯例,-D 为正)
                - dmu_cmd: 期望滚转角速度指令
        """
        logr = self.logr
        device = self.device
        dtype = self.dtype
        if not isinstance(action, torch.Tensor):
            action = torch.asarray(action, device=device, dtype=dtype)
        nx_cmd, ny_cmd, nz_cmd, dmu_cmd = torch.chunk(action, 4, -1)  # (...,1)

        nx_d = torch.where(nx_cmd < 0, -self._nx_min * nx_cmd, nx_cmd * self._nx_max)
        # affcmb(nx_cmd, self._nx_min, self._nx_max)
        # nx_d = affcmb((nx_cmd + 1) * 0.5, self._nx_min, self._nx_max)  # 期望切向过载
        nz_d = nz_cmd * torch.where(
            nz_cmd < 0, self._nz_down_max, self._nz_up_max
        )  # 期望法向过载
        nz_d = -nz_d  # 该死的惯例
        ny_d = ny_cmd * self._ny_max  # 期望侧向过载
        dot_mu = dmu_cmd * self._dot_mu_max  # 期望滚转角速度
        self._n_w[..., 0:1] = nx_d
        self._n_w[..., 1:2] = ny_d
        self._n_w[..., 2:3] = nz_d
        self._dmu.copy_(dot_mu)
        if self.DEBUG:
            assert (action >= -1).all().item() and (action <= 1).all().item(), (
                "action out of range [-1,1].",
            )
            logr.debug(
                (
                    "id:{}".format(self.acmi_id[0].item()),
                    "nx_cmd:{:.3g}".format(nx_cmd[0].item()),
                    "ny_cmd:{:.3g}".format(ny_cmd[0].item()),
                    "nz_cmd:{:.3g}".format(nz_cmd[0].item()),
                    "dmu_cmd:{:.3g}".format(dmu_cmd[0].item()),
                    "nx:{:.3g}".format(nx_d[0].item()),
                    "ny:{:.3g}".format(ny_d[0].item()),
                    "nz:{:.3g}".format(nz_d[0].item()),
                    "dot_mu:{:.3g}".format(dot_mu[0].item()),
                )
            )

    def run(self, index: _SupportedIndexType | None = None):
        msk = self.proc_index(index)
        logr = self.logr
        t = self.sim_time_s()
        pos_e_next, tas_next, Qew_next, mu_next = self._run_ode(
            dt_s=self.sim_step_size_s,
            t_s=t,
            pos_e=self._pos_e,
            tas=self._tas,
            Qew=self._Q_ew,
            mu=self.mu(),
        )
        # 后处理
        Qew_next = normalize(Qew_next)
        tas_next = torch.clip(tas_next, self._Vmin, self._Vmax)  # 防止过零

        self._pos_e.copy_(pos_e_next)
        self._tas.copy_(tas_next)
        self._Q_ew.copy_(Qew_next)
        self.set_mu(mu_next, msk)

        super().run(msk)
        self._propagate(msk)
        if self.DEBUG:
            logr.debug(
                {
                    "tas": tas_next.view(-1, 1)[0, 0].item(),
                    # "|Qew|": Qew_next[0].norm().item(),
                    "mu": mu_next.view(-1, 1)[0, 0].item(),
                }
            )

    def _propagate(self, index: _SupportedIndexType | None):
        """(reset&step 复用)运动学关键状态(TAS,Qew,mu)->全体缓存状态"""
        index = self.proc_index(index)
        # 姿态一致
        self._ppgt_Qew2rpy_ew(index)

        # 速度一致
        self._ppgt_tas2Vw(index)
        self._ppgt_Vw2Ve(index)

        if self._use_geodetic:
            self._ppgt_z2alt(index)

    def _run_ode(
        self,
        dt_s: float | torch.Tensor,  # 积分步长(不考虑掩码) unit: sec
        t_s: float | torch.Tensor,  # 初始时间 unit: sec
        pos_e: torch.Tensor,  # 初始位置地轴系坐标
        tas: torch.Tensor,  # 初始真空速
        Qew: torch.Tensor,  # 初始地轴系/体轴系四元数
        mu: torch.Tensor,  # 初始滚转角(必要冗余,处理万向节死锁)
    ):
        r"""
        求解运动学关键状态转移, 但不修改本体状态\

        Args:
            ...

        Returns:
            pos_e_next (torch.Tensor): 位置地轴系坐标, shape: (N,3)
            tas_next (torch.Tensor): 真空速, shape: (N,1)
            Qew_next (torch.Tensor): 地轴/风轴四元数, shape: (N,4)
            mu_next (torch.Tensor): 地轴/风轴滚转角, shape: (N,1)
        """
        dt_s = dt_s * self.is_alive()
        pos_e_next, tas_next, Qew_next, mu_next = ode_rk45(
            self._f, t_s, (pos_e, tas, Qew, mu), dt_s
        )
        return pos_e_next, tas_next, Qew_next, mu_next

    def _f(self, t, X):
        """动力学"""
        p_e, V, Qew, mu = X

        _0 = self._0F
        n_w = self._n_w  # @ode
        dmu = self._dmu  # @ode
        g = self._g

        a_w = g * n_w  # 过载加速度风轴分量
        if self.use_gravity:  # 考虑重力
            Qwe = quat_conj(Qew)
            a_w += quat_rotate(Qwe, self.g_e())

        # 旋转角速度
        V = torch.clip(V, self._Vmin, self._Vmax)  # 防止过零
        dot_V, a_wy, a_wz = torch.chunk(a_w, 3, dim=-1)
        Vinv = 1 / V
        _P = dmu
        _Q = -a_wz * Vinv
        _R = a_wy * Vinv
        dot_Qew = quat_mul(Qew, 0.5 * torch.cat([_0, _P, _Q, _R], -1))
        dot_p_e = quat_rotate(Qew, torch.cat([V, _0, _0], -1))  # 惯性速度

        dotX = [dot_p_e, dot_V, dot_Qew, dmu]
        return dotX
