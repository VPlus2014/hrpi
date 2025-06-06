from __future__ import annotations
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base_missile import BaseMissile, _SupportedIndexType
    from .base_missile import BaseModel
import torch
import numpy as np

from collections.abc import Sequence
from .base_missile import BaseMissile
from ...utils.math_torch import affcmb, ode_rk23, ode_rk45, ode_euler, quat_conj, modin

_DEBUG = True

# from .base_aircraft import BaseMissile
from environments.utils.math_torch import (
    quat_rotate_inv,
    normalize,
    Qx,
    Qy,
    Qz,
    quat_rotate,
    quat_mul,
    rpy2quat,
)


class PDOF6Missile(BaseMissile):

    def __init__(
        self,
        nyz_max: float = 30,
        Vmin: float = 100,
        Vmax: float = 1000,
        det_rmax: float = 50e3,
        det_fov_deg: float = 120,
        trk_fov_deg: float = 10,
        use_eb=False,
        use_ew=True,
        use_wb=False,
        use_geodetic=True,
        use_mass=True,
        use_inertia=False,
        use_gravity=True,
        **kwargs,
    ):
        """
        伪DOF6 PNG 导弹组

        Args:
            nyz_max (float, optional): 侧向&法向过载限制. Defaults to 30.
            Vmin (float, optional): 最小速度. Defaults to 100.
            Vmax (float, optional): 最大速度. Defaults to 1000.
            det_min (float, optional): 探测半径. Defaults to 50e3.
            det_fov_deg (float, optional): 探测视场FOV, unit degree. Defaults to 120.
            trk_fov_deg (float, optional): 跟踪视场FOV, unit degree. Defaults to 10.
        """
        super().__init__(
            use_eb=use_eb,
            use_ew=use_ew,
            use_wb=use_wb,
            use_mass=use_mass,
            use_geodetic=use_geodetic,
            use_inertia=use_inertia,
            use_gravity=use_gravity,
            **kwargs,
        )
        device = self.device
        dtype = self.dtype

        # simulation parameters
        self._m0 = 84  # initial mass, unit: kg
        self._dm = 6.0  # mass loss rate, unit: kg/s
        self._T = 7063.2  # thrust, unit: N
        self._N = 3.0  # proportionality constant of proportional navigation
        self._nyz_max = nyz_max  # max overload
        self._t_thrust_s = 10  # time limitation of engine, unit: s
        assert self._m0 > self._t_thrust_s * self._dm, (
            "expect m0 > t_thrust_s * dm, got",
            (self._m0, self._t_thrust_s, self._dm),
        )
        self._k_1 = 0.001
        self._K_2 = 0.001
        self._Vmin = Vmin
        self._Vmax = Vmax
        self._dmu_max = math.radians(360 / 1)
        self._dmu_tau = 1e-2
        assert det_rmax > 0, ("det_max should be positive", det_rmax)
        self._det_rmax = det_rmax
        self._det_fov_deg = det_fov_deg
        assert det_fov_deg > 0 and det_fov_deg <= 360, (
            "det_fov_deg should be (0,360],got",
            det_fov_deg,
        )
        self._det_halfa = math.radians(det_fov_deg * 0.5)
        self._det_cosa = math.radians(self._det_halfa)
        assert trk_fov_deg > 0 and trk_fov_deg <= det_fov_deg, (
            "trk_fov_deg should be (0,det_fov_deg],got",
            trk_fov_deg,
        )
        self._trk_halfa = math.radians(trk_fov_deg * 0.5)
        self._trk_cosa = math.radians(self._trk_halfa)

        _0f = self._0f
        _shape = [self.batch_size]

        self._ic_tas = torch.empty(_shape + [1], device=device, dtype=dtype)
        self._ic_rpy_ew = torch.empty(_shape + [3], device=device, dtype=dtype)

        # 状态
        # self._mass = torch.empty_like(_0f)
        self._fake_pos_e = torch.empty(_shape + [3], device=device, dtype=dtype)
        self._fake_vel_e = torch.empty(_shape + [3], device=device, dtype=dtype)
        # 控制量
        self._n_w = torch.zeros(_shape + [3], device=device, dtype=dtype)

    def set_ic_tas(
        self, value: torch.Tensor | float, dst_index: _SupportedIndexType = None
    ):
        """设置发射空速"""
        dst_index = self.proc_batch_index(dst_index)
        self._ic_tas[dst_index] = value + torch.zeros_like(self._ic_tas[dst_index])

    def set_ic_rpy_ew(
        self, value: torch.Tensor | float, dst_index: _SupportedIndexType = None
    ):
        """设置发射姿态"""
        dst_index = self.proc_batch_index(dst_index)
        self._ic_rpy_ew[dst_index, :] = value + torch.zeros_like(
            self._ic_rpy_ew[dst_index, :]
        )

    def reset(self, env_indices: _SupportedIndexType = None):
        env_indices = self.proc_batch_index(env_indices)
        super().reset(env_indices)

        self._tas[env_indices] = self._ic_tas[env_indices]
        self._ppgt_tas2Vw(env_indices)

        self._rpy_ew[env_indices, :] = self._ic_rpy_ew[env_indices, :]
        self._ppgt_rpy_ew2Qew(env_indices)
        self._propagate(env_indices)

    def launch(self, env_indices: _SupportedIndexType = None):
        env_indices = self.proc_batch_index(env_indices)
        self.reset(env_indices)
        super().launch(env_indices)

    def set_ny(self, value: torch.Tensor, dst_index: _SupportedIndexType = None):
        """
        设置侧向过载, unit: G
        """
        dst_index = self.proc_batch_index(dst_index)
        self._n_w[dst_index, 1:2] = value

    def set_nz(self, value: torch.Tensor, dst_index: _SupportedIndexType = None):
        """
        设置法向过载, unit: G
        """
        dst_index = self.proc_batch_index(dst_index)
        self._n_w[dst_index, 2:3] = value

    def run(self):
        t = self.sim_time_s()

        action = self._n_w[..., 1:]

        pos_e_next, tas_next, Qew_next, mu_next = self._run_ode(
            dt_s=self.sim_step_size_s,
            t_s=t,
            action=action,
            pos_e=self._pos_e,
            tas=self._tas,
            Qew=self._Q_ew,
            mu=self.mu(),
            mass=self._mass,
        )
        # 后处理
        super().run()  # time++
        Qew_next = normalize(Qew_next)
        tas_next = torch.clip(tas_next, self._Vmin, self._Vmax)  # 防止过零
        self._pos_e.copy_(pos_e_next)
        self._tas.copy_(tas_next)
        self._Q_ew.copy_(Qew_next)
        self.set_mu(modin(mu_next, -math.pi, math.pi))

        self._propagate()
        super().run()

    def _D(
        self, tas: torch.Tensor, n_y: torch.Tensor, n_z: torch.Tensor
    ) -> torch.Tensor:
        VV = torch.pow(tas, 2).clip(1e-3)
        D_1 = self._k_1 * VV
        D_2 = self._K_2 * (torch.pow(n_y, 2) + torch.pow(n_z, 2)) / VV
        return D_1 + D_2

    def _update_los(
        self, unit_pos: torch.Tensor, unit_vel: torch.Tensor, mask: torch.Tensor
    ):
        """
        计算到全体单位的视线
        Args:
            unit_pos (torch.Tensor): 全体单位位置(NED地轴坐标), shape: (N,3)
            unit_vel (torch.Tensor): 全体单位速度(NED), shape: (N,3)
            mask (torch.Tensor): bool 掩码, shape: (N,M), pos[i] 对导弹 j 是否可用
        """
        assert len(unit_pos.shape) == 2, "units_pos should be 2D tensor"
        assert len(unit_vel.shape) == 2, "units_vel should be 2D tensor"
        assert len(mask.shape) == 2, "mask should be 2D tensor"
        n = unit_pos.shape[0]
        assert unit_vel.shape[0] == n, ("units_vel should have same length as", n)
        m = self.batch_size
        assert mask.shape == (n, m), ("mask should be", (n, m), "got", mask.shape)
        unit_pos = unit_pos.reshape(1, n, -1)
        ego_pos = self._pos_e.reshape(m, 1, -1)
        unit_vel = unit_vel.reshape(1, n, -1)  # (1,n,3)
        los = unit_pos - ego_pos  # (m,n,3)
        ego_vel = self._vel_e.reshape(m, 1, -1)  # (m,1,3)
        dlos = unit_vel - ego_vel  # (m,n,3)
        self._all_los_e = los
        self._all_dlos_e = dlos
        assert mask.shape == (n, m), "mask should be (n,m)"
        assert mask.dtype == torch.bool, "mask should be bool tensor"
        self._los_mask = mask.permute(1, 0).unsqueeze(-1)  # (m,n,1)
        self._los_maskf = self._los_mask.to(dtype=self.dtype)

    def _los2info(self):
        """
        由视线计算其他信息
        """
        dtype = self.dtype
        m = self.batch_size
        e1w_e = quat_rotate(self._Q_ew, self._e1f)  # (m,3)
        e1w_e = e1w_e.reshape(m, 1, -1)  # (m,1,3)
        los_e = self._all_los_e  # (m,n,3)
        dij = torch.norm(los_e, -1, keepdim=True).clip(1e-3)  # (m,n,1)
        cosa = (los_e * e1w_e).sum(-1, keepdim=True) / dij  # (m,n,1)
        mask = self._los_mask
        in_ball = (dij < self._det_rmax) & mask  # 在探测球内
        in_detcone = cosa > self._det_cosa  # 在探测锥内 (m,n,1)
        in_trkcone = cosa > self._trk_cosa  # 在跟踪锥内
        self._candet = candet = in_detcone & in_ball  # 在大视场内 (m,n,1)
        self._cantrk = cantrk = in_trkcone & in_ball  # 在小视场内 (m,n,1)
        self._ndet = self._candet.sum(-2, True)  # 大市场的目标数 (m,1,1)
        self._ntrk = self._cantrk.sum(-2, True)  # 小视场的目标数 (m,1,1)
        self._anydet = anydet = self._candet.any(-2, True)  # 大视场有目标 (m,1,1)
        self._anytrk = anytrk = self._cantrk.any(-2, True)  # 小视场有目标 (m,1,1)

        w1 = torch.where(anytrk, cantrk, candet)  # 视野过滤(>where) (m,n,1)

        all_dlos = self._all_dlos_e  # (m,n,3)
        ener_V = all_dlos.pow(2).sum(-1, keepdim=True)  # (m,n,1)

        w2 = (w1 * ener_V).to(dtype=dtype)  # (m,n,1)
        w2sum = w2.sum(-2, keepdim=True)  # (m,1,1)
        w2_ = torch.where(anydet, w2 / w2sum, 0)  # (m,n,1)
        all_los = self._all_los_e  # (m,n,3)
        tgt_los = (w2_ * all_los).sum(-2)  # (m,3)
        tgt_dlos = (w2_ * all_dlos).sum(-2)  # (m,3)

        Qew = self._Q_ew
        fake_los = quat_rotate(Qew, self._e1f * self._det_rmax)  # (m,3)
        fake_vel = self._vel_e  # (m,3)

        _anytrk = anytrk.reshape(m, 1)  # (m,1)
        tgt_los = torch.where(_anytrk, tgt_los, fake_los)  # (m,3)
        tgt_dlos = torch.where(_anytrk, tgt_dlos, fake_vel)  # (m,3)
        assert tgt_los.shape == (m, 3), "tgt_los should be (m,3)"
        assert tgt_dlos.shape == (m, 3), "tgt_dlos should be (m,3)"

        self._tgt_pos = tgt_los - self._pos_e  # (m,3)
        self._tgt_vel = tgt_dlos - self._vel_e  # (m,3)

        dij = tgt_los.norm(dim=-1, keepdim=True)  # (m,1)
        self._update_distance(dij)

    def observe(
        self,
        unit_pos: torch.Tensor,
        unit_vel: torch.Tensor,
        mask: torch.Tensor,
        auto_control: bool = True,
    ):
        """
        输入潜在目标信息

        目前只支持同一局部坐标系下的目标搜索
        Args:
            unit_pos (torch.Tensor): 单位位置(NED地轴坐标), shape: (N,3)
            unit_vel (torch.Tensor): 单位速度(NED地轴坐标), shape: (N,3)
            mask (torch.Tensor): 掩码, shape: (N,B), 目标 i 信息对导弹 j 是否可用
            auto_control (bool, optional): 1->解算PNG控制量, 0->不改变控制量. Defaults to True.
        """
        assert len(unit_pos.shape) == 2, "unit_pos should be 2D tensor"
        assert len(unit_vel.shape) == 2, "unit_vel should be 2D tensor"
        assert len(mask.shape) == 2, "mask should be 2D tensor"
        self._update_los(unit_pos, unit_vel, mask)
        self._los2info()  # 更新目标信息
        if auto_control:
            nyz = self.png(self._tgt_pos, self._tgt_vel)  # (B,2)
            self._n_w[..., 1:3] = nyz
        return

    def detected_target_num(self, index: _SupportedIndexType = None) -> torch.Tensor:
        """大视场内的物体数

        Args:
            index (_SupportedIndexType, optional): _description_. Defaults to None.

        Returns:
            torch.Tensor: _description_
        """
        index = self.proc_batch_index(index)
        return self._ndet[index]

    def target_position_e(self, index: _SupportedIndexType = None) -> torch.Tensor:
        """感知到的目标位置, shape: (B,3)"""
        index = self.proc_batch_index(index)
        return self._tgt_pos[index] - self.position_e(index)

    def target_velocity_e(self, index: _SupportedIndexType = None) -> torch.Tensor:
        """感知到的目标速度, shape: (B,3)"""
        index = self.proc_batch_index(index)
        return self._tgt_vel[index]

    def _calc_mass(self, index: _SupportedIndexType = None) -> torch.Tensor:
        index = self.proc_batch_index(index)
        return (
            self._m0
            - torch.clip(self.sim_time_s(index), max=self._t_thrust_s) * self._dm
        )

    def _propagate(self, index: _SupportedIndexType = None):
        """(reset&step 复用)运动学关键状态(TAS,Qew,mu)->全体缓存状态"""

        index = self.proc_batch_index(index)
        self._mass[index] = self._calc_mass(index)

        # 姿态一致
        self._ppgt_Qew2rpy_ew(index)

        # 速度一致
        self._ppgt_tas2Vw(index)
        self._ppgt_Vw2Ve(index)

        if self._use_geodetic:
            self._ppgt_z2alt(index)

    def _run_ode(
        self,
        dt_s: torch.Tensor | float,  # 积分步长(不考虑掩码) unit: sec
        t_s: torch.Tensor,  # 初始时间 unit: sec
        action: torch.Tensor,
        pos_e: torch.Tensor,  # 初始位置地轴系坐标
        tas: torch.Tensor,  # 初始真空速
        Qew: torch.Tensor,  # 初始地轴系/体轴系四元数
        mu: torch.Tensor,  # 初始滚转角(必要冗余,处理万向节死锁)
        mass: torch.Tensor,  # 初始质量
    ):
        r"""求解运动学关键状态转移, 但不修改本体状态\
        控制量约定:\
            ny_cmd 为侧向过载\
            nz_cmd 为法向过载系数(NED -Z 为正)\

        Args:
            ...

        Returns:
            pos_e_next (torch.Tensor): 位置地轴系坐标, shape: (B,3)
            tas_next (torch.Tensor): 真空速, shape: (B,1)
            Qew_next (torch.Tensor): 地轴/风轴四元数, shape: (B,4)
        """
        use_gravity = self.use_gravity  # 考虑重力
        use_overloading = True  # 输出过载
        ode_solver = ode_rk23
        logr = self.logr

        ny, nz = torch.split(action, [1, 1], -1)  # (...,1)

        _0f = self._0f
        g = self._g

        if _DEBUG:
            logr.debug(
                {
                    "ny_cmd": ny[0].item(),
                    "nz_cmd": nz[0].item(),
                }
            )

        def _f(
            t: torch.Tensor,
            X: Sequence[torch.Tensor],
        ):
            """动力学"""
            pln = self  # 用于 debug

            p_e, tas, Qew, mu = X

            mask = t < self._t_thrust_s  # 是否推进
            mass = self._calc_mass()  # 质量
            T = (_0f + self._T) * mask

            D = self._D(tas, ny, nz)  # (...,1)
            nx = (T - D) / mass + _0f
            n_w = torch.cat([nx, ny, nz], -1)

            a_w = g * n_w  # 过载加速度风轴分量
            if use_gravity:
                Qwe = quat_conj(Qew)
                a_w += quat_rotate(Qwe, self.g_e())

            # 旋转角速度
            dot_mu = -mu / self._dmu_tau
            tas = torch.clip(tas, self._Vmin, self._Vmax)  # 防止过零
            Vinv = 1 / tas
            dot_tas, a_vy, a_vz = torch.split(a_w, [1, 1, 1], dim=-1)
            omega_w = torch.cat([dot_mu, -a_vz * Vinv, a_vy * Vinv], -1)
            dot_Qew = quat_mul(Qew, torch.cat([_0f, 0.5 * omega_w], -1))

            dot_p_e = quat_rotate(Qew, torch.cat([tas, _0f, _0f], -1))  # 惯性速度

            if _DEBUG:
                logr.debug(
                    {
                        "t": t[0].item(),
                        "nx": nx[0].item(),
                        "D": D[0].item(),
                        "T": T[0].item(),
                    }
                )

            dotX = [dot_p_e, dot_tas, dot_Qew, dot_mu]
            return dotX

        dt_s = dt_s * self.is_launch()
        pos_e_next, tas_next, Qew_next, mu_next = ode_solver(
            _f, (pos_e, tas, Qew, mu), t_s, dt_s
        )
        if _DEBUG:
            logr.debug(
                {
                    "obj": self.__class__.__name__,
                    "id": self.id[0].item(),
                    "tas": tas[0].item(),
                    "|Qew|": Qew_next[0].norm().item(),
                    "mu": mu_next[0].item(),
                    "pos_e": pos_e_next[0].cpu().tolist(),
                    "vel_e": tas_next[0].cpu().tolist(),
                }
            )
        return pos_e_next, tas_next, Qew_next, mu_next

    def png(
        self,
        target_position_e: torch.Tensor,  # (B,3)
        target_velocity_e: torch.Tensor,  # (B,3)
    ) -> torch.Tensor:
        """目标信息->需求控制量"""
        assert target_position_e.shape == (self.batch_size, 3)
        assert target_velocity_e.shape == (self.batch_size, 3)
        dp = target_position_e - self.position_e()  # LOS (B,3)
        dv = target_velocity_e - self.velocity_e()  # relative velocity (B,3)
        dem = torch.norm(dp, dim=-1, keepdim=True).clip(min=1e-3)
        omega = torch.cross(dp, dv, dim=-1) / dem
        ad_e = self._N * torch.cross(
            dv, omega, dim=-1
        )  # required acceleration in earth frame (...,3)
        ad_b = quat_rotate_inv(self.Q_ew(), ad_e)
        nyz_d = ad_b[..., 1:] / self._g
        nyz_d = torch.clip(
            nyz_d,
            min=-self._nyz_max,
            max=self._nyz_max,
        )  # 过载限制 (...,2)
        return nyz_d
