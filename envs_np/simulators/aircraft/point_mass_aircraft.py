from __future__ import annotations
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..proto4model import SupportedIndexType
# import torch
import numpy as np
from typing import Literal
from collections.abc import Sequence
from .base_aircraft import BaseModel, BaseAircraft

_DEBUG = True

# from .base_aircraft import BaseMissile
from ...utils.math_np import (
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
    ndarray,
)

_PI = math.pi


class PointMassAircraft(BaseAircraft):
    def __init__(
        self,
        tas: ndarray,
        rpy_ew: ndarray | float = 0,
        alpha: ndarray | float = 0,
        rho=1.29,
        **kwargs,
    ) -> None:
        """引入迎角控制的伪DOF6刚体飞机(无转动惯量 & 小迎角&无侧滑)

        Args:
            tas (NDArr|float): 初始真空速 unit: m/s, shape: (B, 3), default: 0
            rpy_ew (NDArr|float): 初始速度系姿态欧拉角(mu,gamma,chi) unit: rad, shape: (B, 3), default: 0
            alpha (NDArr|float): 初始迎角 unit: rad, shape: (B, 1), default: 0
            **kwargs: 其他参数, 见 BaseAircraft
        """
        super().__init__(
            use_eb=True,
            use_ew=True,
            use_wb=True,
            **kwargs,
        )
        device = self.device
        dtype = self.dtype
        bsz = self.batch_size

        # simulation parameters
        self._m = 7500  # aircraft mass, unit: kg
        self._S = 26.0  # 翼面 reference area, unit: m^2
        self._C_L_max = 0.753  #
        self._c_L_alpha = 4.01  # \partial{C_L}/\partial{\alpha}
        self._c_D_0 = 0.0169  # 零升阻力系数
        self._k_D = 0.179  # 升致阻力因子
        self._a_tmax = 7.0  #
        self._alpha_max = math.radians(15)  # 最大迎角(安全阈值<=15 deg), unit: rad
        self._rho = 1.29  # atmosphere density, unit: kp/m^3
        #
        self._tau_mu = 2.0  # 滚转响应 惯性时间常数 unit: sec
        self._tau_alpha = 2.0  # 迎角响应 惯性时间常数 unit: sec
        self._dot_mu_max = math.radians(360 / 3)  # 3 秒一圈
        self._dot_alpha_max = math.radians(5)
        #
        G = self._m * self._g
        self._T_max = G * 2.5  # (加力) max thrust, unit: N
        self._T_min = G * 0.5  # (慢车) min thrust, unit: N

        # initial conditions
        _0 = np.zeros((bsz, 1), device=device, dtype=dtype)
        self._ic_rpy_ew = np.empty((bsz, 3), device=device, dtype=dtype)
        self._ic_tas = np.empty((bsz, 1), device=device, dtype=dtype)
        self._ic_alpha = np.empty((bsz, 1), device=device, dtype=dtype)

        self._ic_rpy_ew.copy_(rpy_ew + np.zeros_like(self._ic_rpy_ew))
        self._ic_tas.copy_(tas + _0)
        self._ic_alpha.copy_(alpha + _0)

        # 控制量
        self._Td = np.empty_like(_0)  # 期望推力
        self._alpha_d = np.empty_like(_0)  # 期望迎角
        self._mu_d = np.empty_like(_0)  # 期望滚转角

        # 过载
        self._nx = np.zeros((bsz, 1), device=device, dtype=dtype)
        self._ny = np.zeros((bsz, 1), device=device, dtype=dtype)
        self._nz = np.zeros((bsz, 1), device=device, dtype=dtype)

    @property
    def q_kg(self):
        return self.Q_ew()

    def reset(
        self,
        env_indices: SupportedIndexType | None,
    ):
        env_indices = self.proc_index(env_indices)

        super().reset(env_indices)
        self._rpy_ew[env_indices] = self._ic_rpy_ew[env_indices]
        self._tas[env_indices] = self._ic_tas[env_indices]
        self.set_alpha(self._ic_alpha[env_indices], env_indices)

        self._ppgt_rpy_ew2Qew(env_indices)
        self._propagate(env_indices)
        pass

    def set_action(self, action: np.ndarray | ndarray):
        """
        控制量约定: 全部在 [-1,1] 内!
        """
        logr = self.logger
        if not isinstance(action, ndarray):
            action = ndarray(action, device=self.device, dtype=self.dtype)

        thrust_cmd, alpha_cmd, mu_cmd = np.chunk(action, 3, axis=-1)  # (...,1)

        mu_d = mu_cmd * _PI  # 期望滚转角
        alpha_d = affcmb(
            -self._alpha_max, self._alpha_max, 2 * alpha_cmd + 1
        )  # 期望迎角
        thrust_d = affcmb(self._T_min, self._T_max, 2 * thrust_cmd + 1)  # 期望推力

        self._Td.copy_(thrust_d)
        self._alpha_d.copy_(alpha_d)
        self._mu_d.copy_(mu_d)

        if _DEBUG:
            logr.debug(
                {
                    "thrust_cmd": thrust_cmd[0].item(),
                    "alpha_cmd": alpha_cmd[0].item(),
                    "mu_cmd": mu_cmd[0].item(),
                    "thrust_d": thrust_d[0].item(),
                    "alpha_d": alpha_d[0].item(),
                    "mu_d": mu_d[0].item(),
                }
            )

    def run(self):
        t = self.sim_time_s()
        pos_e_next, tas_next, Qew_next, mu_next, alpha_next = self._run_ode(
            dt_s=self.sim_step_size_s,
            t_s=t,
            pos_e=self._pos_e,
            tas=self._tas,
            Qew=self._Q_ew,
            mu=self.mu(),
            alpha=self.alpha(),
        )
        # 后处理
        Qew_next = normalize(Qew_next)

        self._pos_e.copy_(pos_e_next)
        self._tas.copy_(tas_next)
        self._Q_ew.copy_(Qew_next)
        self.set_mu(mu_next)
        self.set_alpha(alpha_next)

        self._propagate()
        super().run()

    def _propagate(self):
        """(reset&step 复用)运动学关键状态(TAS,Qew,mu,alpha)->全体缓存状态"""
        # 姿态一致
        self._ppgt_rpy_wb2Qwb()
        self._ppgt_QewQwb_to_Qeb()
        ## 欧拉角
        self._ppgt_Qeb2rpy_eb()
        self._ppgt_Qew2rpy_ew()

        # 速度一致
        self._ppgt_tas2Vw()
        self._ppgt_Vw2Vb()
        self._ppgt_Vw2Ve()

    def _run_ode(
        self,
        dt_s: float | ndarray,  # 积分步长(不考虑掩码) unit: sec
        t_s: float | ndarray,  # 初始时间 unit: sec
        pos_e: ndarray,  # 初始位置地轴系坐标
        tas: ndarray,  # 初始真空速
        Qew: ndarray,  # 初始地轴系/体轴系四元数
        mu: ndarray,  # 初始滚转角(必要冗余,处理万向节死锁)
        alpha: ndarray,  # 初始迎角
    ):
        r"""求解运动学关键状态转移, 但不修改本体状态\
        控制量约定:\
            thrust_cmd 为归一化推力系数 in [0,1]\
            alpha_cmd 为归一化期望迎角系数 in [0,1]\
            mu_cmd 为归一化期望滚转角系数 in [0,1]

        Args:
            ...

        Returns:
            pos_e_next (NDArr): 位置地轴系坐标, shape: (B,3)
            tas_next (NDArr): 真空速, shape: (B,1)
            Qew_next (NDArr): 地轴/风轴四元数, shape: (B,4)
            mu_next (NDArr): 地轴/风轴滚转角, shape: (B,1)
            alpha_next (NDArr): 迎角, shape: (B,1)
        """
        use_gravity = True  # 考虑重力
        use_overloading = True  # 输出过载
        ode_solver = ode_rk23

        _0 = self._0F
        _1 = self._1F
        mu_d = self._mu_d
        alpha_d = self._alpha_d
        thrust_d = self._Td

        def _f(
            t: ndarray,
            X: Sequence[ndarray],
        ):
            """动力学"""
            pln = self  # 用于 debug

            p_e, tas, Qew, mu, alpha = X

            dot_mu = delta_rad_reg(mu_d, mu) / self._tau_mu  # -> mu_d
            dot_alpha = (alpha_d - alpha) / self._tau_alpha  # -> alpha_d
            dot_mu = np.clip(dot_mu, -self._dot_mu_max, self._dot_mu_max)
            dot_alpha = np.clip(dot_alpha, -self._dot_alpha_max, self._dot_alpha_max)

            Qwe = quat_conj(Qew)
            Qwb = Qy(alpha)
            Qbw = quat_conj(Qwb)

            F_b = _0.clone()  # 过载合力(体轴) (B,3)

            T = thrust_d  # 推力 (B,1)
            F_b[..., 0:1] += T

            # 气动力
            qbar = self.qbar(tas, self._rho)
            qS = qbar * self._S  # 动压面积
            ##
            L = self.L_a(alpha, qS)  # 升力
            F_b[..., 2:3] -= L
            ##
            D = self.D_a(alpha, qS)  # 阻力
            D_w = np.cat([-D, _0, _0], axis=-1)
            F_b += quat_rotate(Qbw, D_w)

            a_b = F_b / self._m  # 过载加速度体轴分量
            if use_overloading:
                nx, ny, nz = np.split(a_b / self._g, [1, 1, 1], axis=-1)  # 输出过载
                self._nx.copy_(nx)
                self._ny.copy_(ny)
                self._nz.copy_(-nz)
            if use_gravity:
                a_b += quat_rotate(Qwe, self.g_e())

            a_w = quat_rotate(Qwb, a_b)  # 惯性加速度风轴分量

            # 旋转角速度
            tas = np.clip(tas, 1e-3)  # 防止过零
            Vinv = 1 / tas
            dot_tas, a_vy, a_vz = np.split(a_w, [1, 1, 1], axis=-1)
            omega_w = np.cat([dot_mu, -a_vz * Vinv, a_vy * Vinv], axis=-1)
            dot_Qew = quat_mul(Qew, np.cat([_0, 0.5 * omega_w], axis=-1))

            dot_p_e = quat_rotate(Qew, np.cat([tas, _0, _0], axis=-1))  # 惯性速度

            dotX = [dot_p_e, dot_tas, dot_Qew, dot_mu, dot_alpha]
            return dotX

        dt_s = dt_s * self.is_alive()
        pos_e_next, tas_next, Qew_next, mu_next, alpha_next = ode_solver(
            _f, (pos_e, tas, Qew, mu, alpha), t_s, dt_s
        )
        if _DEBUG:
            logr.debug(
                {
                    "tas": tas[0].item(),
                    # "|Qew|": Qew[0].norm().item(),
                    "mu": mu[0].item(),
                    "alpha": alpha[0].item(),
                    "nx": self._nx[0].item(),
                    "ny": self._ny[0].item(),
                    "nz": self._nz[0].item(),
                }
            )
        return pos_e_next, tas_next, Qew_next, mu_next, alpha_next

    # 空气动力计算
    def c_L(self, alpha: ndarray) -> ndarray:
        """升力系数模型"""
        c_L = self._c_L_alpha * alpha  # 小迎角线性模型
        return c_L

    def L_a(self, alpha: ndarray, qS: ndarray) -> ndarray:
        r"""升力模型, qS=\bar{q}*S"""
        L_a = qS * self.c_L(alpha)
        return L_a

    def c_D(self, alpha: ndarray) -> ndarray:
        """阻力系数模型"""
        return self._c_D_0 + self._k_D * self.c_L(alpha).pow(2)

    def D_a(self, alpha: ndarray, qS: ndarray) -> ndarray:
        r"""阻力模型, qS=\bar{q}*S"""
        D_a = qS * self.c_D(alpha)
        return D_a

    def qbar(self, tas: ndarray, rho: float) -> ndarray:
        r"""
        动态压力 \bar{q}
        Args:
            tas (NDArr): true air speed, unit: m/s
            rho (float): air density, unit: kg/m^3
        Returns:
            NDArr: dynamic pressure, unit: N/m^2
        """
        return 0.5 * rho * tas.pow(2)
