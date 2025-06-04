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
from ...utils.math import (
    normalize,
    Qx,
    Qy,
    Qz,
    quat_rotate,
    quat_mul,
    quat_conj,
    ode_rk45,
    ode_rk23,
    ode_euler,
    delta_rad_reg,
    affcmb,
    rpy2quat,
    rpy2quat_inv,
    uvw2alpha_beta,
)

_PI = math.pi


class PesudoDOF6(BaseAircraft):
    def __init__(
        self,
        tas: torch.Tensor,
        rpy_ew: torch.Tensor | float = 0,
        nx_max: float = 0.0,
        nx_min: float = -0.5,
        ny_max: float = 0.5,
        nz_max: float = 8.0,
        nz_min: float = -0.5,
        Vmin: float = 240,
        Vmax: float = 240,
        dmu_max: float = math.radians(360 / 6),
        use_gravity: bool = False,
        **kwargs,
    ) -> None:
        """伪DOF6刚体飞机(无转动惯量 & 过载控制 & 只有速度系)

        Args:
            tas (torch.Tensor|float): 初始真空速 unit: m/s, shape: (B, 3), default: 0
            rpy_ew (torch.Tensor|float): 初始速度系姿态欧拉角(mu,gamma,chi) unit: rad, shape: (B, 3), default: 0
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
        bsz = self.batchsize

        assert ny_max >= 0, ("ny_max must be non-negative.", ny_max)
        assert nx_max >= nx_min, (
            "nx_max must be greater than or equal to nx_min.",
            nx_max,
            nx_min,
        )
        assert nz_max >= nz_min, (
            "nz_max must be greater than or equal to nz_min.",
            nz_max,
            nz_min,
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
        self._nx_min = nx_min
        self._ny_max = ny_max
        self._ny_min = -ny_max
        self._nz_max = nz_max
        self._nz_min = nz_min
        self._Vmin = Vmin
        self._Vmax = Vmax
        self._dot_mu_max = dmu_max

        # initial conditions
        _0 = torch.zeros((bsz, 1), device=device, dtype=dtype)
        self._ic_rpy_ew = torch.empty((bsz, 3), device=device, dtype=dtype)
        self._ic_tas = torch.empty((bsz, 1), device=device, dtype=dtype)

        self._ic_rpy_ew.copy_(rpy_ew + torch.zeros_like(self._ic_rpy_ew))
        self._ic_tas.copy_(tas + _0)

        # 当前控制量
        self._n_w = torch.zeros((bsz, 3), device=device, dtype=dtype)
        self._dmu = torch.zeros((bsz, 1), device=device, dtype=dtype)

    def reset(
        self,
        env_indices: _SupportedIndexType = None,
    ):
        env_indices = self.proc_batch_index(env_indices)

        super().reset(env_indices)
        self._rpy_ew[env_indices] = self._ic_rpy_ew[env_indices]
        self._tas[env_indices] = self._ic_tas[env_indices]

        self._ppgt_rpy_ew2Qew(env_indices)
        self._propagate(env_indices)
        pass

    def run(self, action: np.ndarray | torch.Tensor):
        """
        控制量约定: 全部在 [0,1] 内!
        """
        device = self.device
        dtype = self.dtype
        if not isinstance(action, torch.Tensor):
            action = torch.asarray(action, device=device, dtype=dtype)

        t = self.sim_time_s
        pos_e_next, tas_next, Qew_next, mu_next = self._run_ode(
            dt_s=self.sim_step_size_s,
            t_s=t,
            action=action,
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
        self.set_mu(mu_next)

        self._propagate()
        super().run(action)

    def _propagate(self, index: _SupportedIndexType = None):
        """(reset&step 复用)运动学关键状态(TAS,Qew,mu)->全体缓存状态"""
        index = self.proc_batch_index(index)
        # 姿态一致
        self._ppgt_Qew2rpy_ew(index)

        # 速度一致
        self._ppgt_tas2Vw(index)
        self._ppgt_Vw2Ve(index)

    def _run_ode(
        self,
        dt_s: float | torch.Tensor,  # 积分步长(不考虑掩码) unit: sec
        t_s: float | torch.Tensor,  # 初始时间 unit: sec
        action: torch.Tensor,
        pos_e: torch.Tensor,  # 初始位置地轴系坐标
        tas: torch.Tensor,  # 初始真空速
        Qew: torch.Tensor,  # 初始地轴系/体轴系四元数
        mu: torch.Tensor,  # 初始滚转角(必要冗余,处理万向节死锁)
    ):
        r"""求解运动学关键状态转移, 但不修改本体状态\
        控制量约定:\
            nx_cmd 为归一化切向过载系数 in [0,1]\
            ny_cmd 为归一化侧向过载系数 in [0,1]\
            nz_cmd 为归一化法向过载系数(NED -Z 为正) in [0,1]\
            dmu_cmd 为归一化滚转角速度系数 in [0,1]\

        Args:
            ...

        Returns:
            pos_e_next (torch.Tensor): 位置地轴系坐标, shape: (B,3)
            tas_next (torch.Tensor): 真空速, shape: (B,1)
            Qew_next (torch.Tensor): 地轴/风轴四元数, shape: (B,4)
            mu_next (torch.Tensor): 地轴/风轴滚转角, shape: (B,1)
        """
        use_gravity = self.use_gravity  # 考虑重力
        use_overloading = True  # 输出过载
        ode_solver = ode_rk45
        logr = self.logr

        nx_cmd, ny_cmd, nz_cmd, dmu_cmd = torch.split(
            action, [1, 1, 1, 1], -1
        )  # (...,1)

        nx_d = affcmb(nx_cmd, self._nx_min, self._nx_max)  # 期望切向过载
        ny_d = affcmb(ny_cmd, self._ny_min, self._ny_max)  # 期望横向过载
        nz_d = affcmb(nz_cmd, self._nz_min, self._nz_max)  # 期望法向过载
        dot_mu = affcmb(dmu_cmd, -_PI, _PI)  # 期望滚转角
        self._n_w[..., 0:1] = nx_d
        self._n_w[..., 1:2] = ny_d
        self._n_w[..., 2:3] = -nz_d
        self._dmu.copy_(dot_mu)

        if _DEBUG:
            logr.debug(
                {
                    "nx_cmd": nx_cmd[0].item(),
                    "ny_cmd": ny_cmd[0].item(),
                    "nz_cmd": nz_cmd[0].item(),
                    "dmu_cmd": dmu_cmd[0].item(),
                }
            )

        _0 = self._0f
        g = self._g

        def _f(
            t: torch.Tensor,
            X: Sequence[torch.Tensor],
        ):
            """动力学"""
            pln = self  # 用于 debug

            p_e, tas, Qew, mu = X

            Qwe = quat_conj(Qew)

            a_w = g * self._n_w  # 过载加速度风轴分量
            if use_gravity:
                a_w += quat_rotate(Qwe, self.g_e)

            # 旋转角速度
            tas = torch.clip(tas, self._Vmin, self._Vmax)  # 防止过零
            Vinv = 1 / tas
            dot_tas, a_vy, a_vz = torch.split(a_w, [1, 1, 1], dim=-1)
            omega_w = torch.cat([dot_mu, -a_vz * Vinv, a_vy * Vinv], -1)
            dot_Qew = quat_mul(Qew, torch.cat([_0, 0.5 * omega_w], -1))

            dot_p_e = quat_rotate(Qew, torch.cat([tas, _0, _0], -1))  # 惯性速度

            dotX = [dot_p_e, dot_tas, dot_Qew, dot_mu]
            return dotX

        dt_s = dt_s * self.is_alive()
        pos_e_next, tas_next, Qew_next, mu_next = ode_solver(
            _f, (pos_e, tas, Qew, mu), t_s, dt_s
        )
        if _DEBUG:
            logr.debug(
                {
                    "tas": tas[0].item(),
                    "|Qew|": Qew[0].norm().item(),
                    "mu": mu[0].item(),
                    "nx": self._n_w[0, 0].item(),
                    "ny": self._n_w[0, 1].item(),
                    "nz": -(self._n_w[0, 2].item()),
                }
            )
        return pos_e_next, tas_next, Qew_next, mu_next
