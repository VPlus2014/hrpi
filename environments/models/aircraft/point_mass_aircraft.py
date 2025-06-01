from __future__ import annotations
import math
from typing import TYPE_CHECKING
import torch
import numpy as np
from typing import Literal
from collections.abc import Sequence
from .base_aircraft import BaseModel, BaseAircraft

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
)

_PI = math.pi


class PointMassAircraft(BaseAircraft):
    def __init__(
        self,
        vel_b: torch.Tensor | float = 0,
        rpy_eb: torch.Tensor | float = 0,
        **kwargs,
    ) -> None:
        """伪DOF6刚体飞机(无转动惯量&有迎角&无侧滑)

        Args:
            vel_b (torch.Tensor|float): 初始惯性速度体轴系分量 unit: m/s, shape: (B, 3), default: 0
            rpy_eb (torch.Tensor|float): 初始体轴系姿态(欧拉角) unit: rad, shape: (B, 3), default: 0
        """
        super().__init__(use_body_frame=True, **kwargs)
        device = self.device
        dtype = self.dtype
        bsz = self.batchsize

        # simulation parameters
        self._m = 7500  # aircraft mass, unit: kg
        self._S = 26  # 翼面 reference area, unit: m^2
        self._C_L_max = 0.753  #
        self._c_L_alpha = 4.01  # \partial{C_L}/\partial{\alpha}
        self._c_D_0 = 0.0169  # 零升阻力系数
        self._k_D = 0.179  # 升致阻力因子
        self._a_tmax = 7.0  #
        self._T_max = 219755  # (加力) max thrust, unit: N
        self._T_min = 0.0  # (慢车) min thrust, unit: N
        self._alpha_max = math.radians(15)  # 最大迎角(安全阈值), unit: rad
        self._tau_mu = 0.3  # 滚转响应 惯性时间常数 unit: sec
        self._tau_alpha = 0.3  # 迎角响应 惯性时间常数 unit: sec

        # initial conditions
        self._ic_rpy_eb = torch.empty((bsz, 3), device=device, dtype=dtype)
        self._ic_vel_b = torch.empty((bsz, 3), device=device, dtype=dtype)
        self._ic_rpy_eb.fill_(rpy_eb)
        self._ic_vel_b.fill_(vel_b)

    @property
    def q_kg(self):
        return self.Q_ea

    def reset(
        self,
        env_indices: Sequence[int] | torch.Tensor | None = None,
    ):
        env_indices = self.proc_indices(env_indices)

        super().reset(env_indices)
        self._rpy_eb[env_indices] = self._ic_rpy_eb[env_indices]
        self._vel_b[env_indices] = self._ic_vel_b[env_indices]

        self._ppgt_rpy_eb2Qeb()
        self.__propagate()
        pass

    def run(self, action: np.ndarray | torch.Tensor):
        device = self.device
        dtype = self.dtype
        if not isinstance(action, torch.Tensor):
            action = torch.asarray(action, device=device, dtype=dtype)

        t = self.sim_time_s
        pos_e_next, vel_b_next, Qeb_next, roll_next = self._run_ode(
            dt_s=self.sim_step_size_s,
            t_s=t,
            action=action,
            pos_e=self.position_e,
            vel_b=self.velocity_b,
            Qeb=self._Q_eb,
            roll=self.roll,
        )

        self._pos_e.copy_(pos_e_next)
        self._vel_b.copy_(vel_b_next)
        self._Q_eb.copy_(Qeb_next)

        self.__propagate()
        super().run(action)

    def __propagate(self):
        """(reset&step 复用)运动学关键状态(UVW,Qeb)->全体缓存状态"""
        self._ppgt_vb2tas()
        self._ppgt_vb2ve()

        self._ppgt_Qeb2rpy_eb()

        self._ppgt_vb2rpy_ba()
        self._ppgt_rpy_ba2Qba()

        self._ppgt_Qea()
        self._ppgt_Qea2rpy_ea()

    def _run_ode(
        self,
        dt_s: float | torch.Tensor,  # 积分步长(不考虑掩码) unit: sec
        t_s: float | torch.Tensor,  # 初始时间 unit: sec
        action: torch.Tensor,
        pos_e: torch.Tensor,  # 初始位置地轴系坐标
        vel_b: torch.Tensor,  # 初始惯性速度体轴系坐标
        Qeb: torch.Tensor,  # 初始地轴系/体轴系四元数
        roll: torch.Tensor,  # 初始体轴系滚转角(必要冗余,处理万向节死锁)
    ):
        r"""求解运动学关键状态转移, 但不修改本体状态\
        控制量约定:\
            thrust_cmd 为归一化推力系数 in [0,1]\
            alpha_cmd 为归一化期望迎角系数 in [-1,1]\
            mu_cmd 为归一化期望滚转角系数 in [-1,1]

        Args:
            ...

        Returns:
            pos_e_next: 位置\
            vel_b_next: 速度\
            Qeb_next: 姿态\
            roll_next: 滚转角\
        """
        use_gravity = True  # 考虑重力
        use_overloading = True  # 输出过载
        solver = ode_rk23

        thrust_cmd, alpha_cmd, mu_cmd = torch.split(
            action, [1, 1, 1], dim=-1
        )  # (...,1)

        mu_d = mu_cmd * _PI  # 期望滚转角
        alpha_d = alpha_cmd * self._alpha_max  # 期望迎角
        thrust_d = affcmb(thrust_cmd, self._T_min, self._T_max)  # 期望推力

        _0_1d = torch.zeros_like(self._tas)
        _1_1d = torch.ones_like(_0_1d)
        _e1 = torch.cat([_0_1d, _1_1d, _0_1d], -1)

        buf_roll = roll.clone()

        def _f(
            t: torch.Tensor,
            X: Sequence[torch.Tensor],
        ):
            """动力学"""
            p_e, uvw, Qeb = X
            nonlocal buf_roll
            Qeb = normalize(Qeb)  # 规范化(大步长下必要操作)
            Qbe = quat_conj(Qeb)
            buf_roll = rpy2quat_inv(Qeb, buf_roll)[..., 0:1]
            tas = torch.norm(uvw, p=2, dim=-1, keepdim=True)
            eUVW = torch.where(tas < 1e-2, _e1, uvw / tas)  # 速度系X轴的体轴坐标
            alpha = -torch.asin(eUVW[..., 2:3])  # 迎角

            F_b = torch.zeros_like(uvw)  # 过载合力

            T = thrust_d  # 推力
            F_b[..., 0:1] += T

            # 气动力
            qbar = self.qbar(tas, self._rho)
            qS = qbar * self._S  # 动压面积
            L = self.L_a(alpha, qS)
            F_b[..., 2:3] -= L  # 升力

            D = self.D_a(alpha, qS)  # 阻力
            F_b -= eUVW * D

            a_b = F_b / self._m  # 惯性加速度体轴分量
            if use_overloading:
                nx, ny, nz = torch.split(a_b / self._g, [1, 1, 1], dim=-1)  # 输出过载

            if use_gravity:
                a_b += quat_rotate(Qbe, self.g_e)
            a_e = quat_rotate(Qeb, a_b)  # 惯性加速度地轴分量

            # 旋转角速度
            dot_mu = delta_rad_reg(mu_d, buf_roll) / self._tau_mu
            dot_alpha = (alpha_d - alpha) / self._tau_alpha
            PQR = torch.cat([dot_mu, dot_alpha, _0_1d], -1)
            dot_Qeb = quat_mul(Qeb, torch.cat([_0_1d, 0.5 * PQR], -1))

            dot_uvw = a_e - PQR.cross(uvw, -1)

            dot_p_e = quat_rotate(Qeb, uvw)  # 惯性速度地轴分量
            return dot_p_e, dot_uvw, dot_Qeb

        dt_s = dt_s * self.is_alive()
        pos_e_next, vel_b_next, Qeb_next = solver(_f, (pos_e, vel_b, Qeb), t_s, dt_s)

        # 后处理
        Qeb_next = normalize(Qeb_next)
        roll_next = rpy2quat_inv(Qeb_next, buf_roll)

        return pos_e_next, vel_b_next, Qeb_next, roll_next

    # 空气动力计算
    def c_L(self, alpha: torch.Tensor) -> torch.Tensor:
        """升力系数模型"""
        c_L = self._c_L_alpha * alpha  # 小迎角线性模型
        return c_L

    def L_a(self, alpha: torch.Tensor, qS: torch.Tensor) -> torch.Tensor:
        """升力模型"""
        L_a = qS * self.c_L(alpha)
        return L_a

    def c_D(self, alpha: torch.Tensor) -> torch.Tensor:
        """阻力系数模型"""
        return self._c_D_0 + self._k_D * self.c_L(alpha).pow(2)

    def D_a(self, alpha: torch.Tensor, qS: torch.Tensor) -> torch.Tensor:
        """阻力模型"""
        D_a = qS * self.c_D(alpha)
        return D_a

    def qbar(self, tas: torch.Tensor, rho: float) -> torch.Tensor:
        r"""
        动态压力 \bar{q}
        Args:
            tas (torch.Tensor): true air speed, unit: m/s
            rho (float): air density, unit: kg/m^3
        Returns:
            torch.Tensor: dynamic pressure, unit: N/m^2
        """
        return 0.5 * rho * tas.pow(2)
