from __future__ import annotations
import math
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from .base_missile import BaseModel, BaseMissile, SupportedMaskType
# import torch
import numpy as np

from collections.abc import Sequence
from .base_missile import BaseMissile
from ...utils.math_np import (
    affcmb,
    ode_rk23,
    ode_rk45,
    ode_euler,
    quat_conj,
    modin,
    bkbn,
    unbind_keepdim,
    split_,
    cat,
    BoolNDArr,
    DoubleNDArr,
    pow,
    normalize,
    unsqueeze,
    quat_rotate,
    quat_mul,
    quat_rotate_inv,
    cross,
    norm_,
    clip,
    asarray,
    sum_,
    ndarray,
    zeros,
    zeros_like,
    where,
)


# from .base_aircraft import BaseMissile


class P6DOFMissile(BaseMissile):

    def __init__(
        self,
        nyz_max: float = 100,
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
        kill_radius=100,
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
            kill_radius (float, optional): 捕获半径. Defaults to 100.
        """
        super().__init__(
            use_eb=use_eb,
            use_ew=use_ew,
            use_wb=use_wb,
            use_mass=use_mass,
            use_geodetic=use_geodetic,
            use_inertia=use_inertia,
            use_gravity=use_gravity,
            kill_radius=kill_radius,
            **kwargs,
        )
        device = self.device
        dtype = self.dtype

        # simulation parameters
        self._m0 = 84  # initial mass, unit: kg
        self._dm = 6.0  # mass loss rate, unit: kg/s
        self._T = 7063.2  # thrust, unit: N
        self._K_PN = 10.0  # proportionality constant of proportional navigation
        self._nyz_max = nyz_max  # max overload
        assert nyz_max > 0, "nyz_max should be positive"
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
        self._dmu_K = 10.0
        assert det_rmax > 0, ("det_max should be positive", det_rmax)
        self._det_rmax = det_rmax
        self._det_fov_deg = det_fov_deg
        assert det_fov_deg > 0 and det_fov_deg <= 360, (
            "det_fov_deg should be (0,360],got",
            det_fov_deg,
        )
        self._det_halfa = math.radians(det_fov_deg * 0.5)
        self._det_cosa = math.cos(self._det_halfa)
        assert trk_fov_deg > 0 and trk_fov_deg <= det_fov_deg, (
            "trk_fov_deg should be (0,det_fov_deg],got",
            trk_fov_deg,
        )
        self._trk_halfa = math.radians(trk_fov_deg * 0.5)
        self._trk_cosa = math.cos(self._trk_halfa)

        _shape = [*self.group_shape]

        self._ic_tas = zeros(
            _shape + [1],
            dtype=dtype,
            # device=device,
        )
        self._ic_rpy_ew = zeros(
            _shape + [3],
            dtype=dtype,
            # device=device,
        )
        self._ic_pos_e = zeros(
            _shape + [3],
            dtype=dtype,
            # device=device,
        )

        # 控制量
        self._n_w = zeros(
            _shape + [3],
            dtype=dtype,
            # device=device,
        )

    def set_ic_tas(self, value: ndarray | float, dst_index: SupportedMaskType | None):
        """设置发射空速"""
        dst_index = self.proc_to_mask(dst_index)
        self._ic_tas[dst_index, :] = value

    def set_ic_rpy_ew(
        self, value: ndarray | float, dst_index: SupportedMaskType | None
    ):
        """设置发射姿态"""
        dst_index = self.proc_to_mask(dst_index)
        self._ic_rpy_ew[dst_index, :] = value

    def set_ic_pos_e(self, value: ndarray | float, dst_index: SupportedMaskType | None):
        """设置初始位置"""
        dst_index = self.proc_to_mask(dst_index)
        self._ic_pos_e[dst_index, :] = value

    def reset(self, mask: SupportedMaskType | None):
        mask = self.proc_to_mask(mask)
        super().reset(mask)

        self._mass[mask, :] = self._m0

        self._tas[mask, :] = self._ic_tas[mask, :]
        self._ppgt_tas2Vw(mask)

        self._rpy_ew[mask, :] = self._ic_rpy_ew[mask, :]
        self._ppgt_rpy_ew2Qew(mask)

        self._pos_e[mask, :] = self._ic_pos_e[mask, :]

        self._propagate(mask)

    def launch(self, mask: SupportedMaskType | None):
        mask = self.proc_to_mask(mask)
        self.reset(mask)
        super().launch(mask)

    def set_ny(self, value: ndarray, dst_index: SupportedMaskType | None):
        """
        设置侧向过载, unit: G
        """
        dst_index = self.proc_to_mask(dst_index)
        self._n_w[dst_index, 1:2] = value

    def set_nz(self, value: ndarray, dst_index: SupportedMaskType | None):
        """
        设置法向过载, unit: G
        """
        dst_index = self.proc_to_mask(dst_index)
        self._n_w[dst_index, 2:3] = value

    def run(self, mask: SupportedMaskType | None):
        msk = self.proc_to_mask(mask)
        t = self.sim_time_s()

        action = self._n_w[..., 1:]
        pos_e_next, tas_next, Qew_next, mu_next, mass_next = self._run_ode(
            dt_s=self.sim_step_size_s,
            t_s=t,
            action=action,
            pos_e=self._pos_e,
            tas=self._tas,
            Qew=self._Q_ew,
            mu=self.mu(),
            mass=self._mass,
            mask=msk,
        )
        # 后处理
        super().run(msk)  # time++
        Qew_next = normalize(Qew_next)
        tas_next = clip(tas_next, self._Vmin, self._Vmax)  # 防止过零
        self._pos_e[mask, :] = pos_e_next[mask, :]
        self._tas[mask, :] = tas_next[mask, :]
        self._Q_ew[mask, :] = Qew_next[mask, :]
        self.set_mu(mu_next[mask, :], msk)
        self._mass[mask, :] = mass_next[mask, :]

        self._propagate(msk)

    def _D(self, tas: ndarray, n_y: ndarray, n_z: ndarray) -> ndarray:
        VV = pow(tas, 2).clip(1e-3)
        D_1 = self._k_1 * VV
        D_2 = self._K_2 * (pow(n_y, 2) + pow(n_z, 2)) / VV
        return D_1 + D_2

    def _update_los(
        self,
        unit_pos: ndarray,
        unit_vel: ndarray,
        unit_id: ndarray,
        mask: BoolNDArr,
    ):
        """
        计算到全体单位的视线
        Args:
            unit_pos (ndarray): 全体单位位置(NED地轴坐标), shape: (...,n,3)
            unit_vel (ndarray): 全体单位速度(NED), shape: (...,n,3)
            unit_id (ndarray): 全体单位ID, shape: (...,n,1)
            mask (ndarray): bool 掩码, shape: (...,n,N), 对象[i] 信息对导弹 j 是否可用
        """
        assert (
            len(unit_pos.shape)
            == len(unit_vel.shape)
            == len(mask.shape)
            == len(self._pos_e.shape)
        ), (
            f"expect unit_pos, unit_vel, mask are {len(self._pos_e.shape)}-D tensor, got",
            (len(unit_pos.shape), len(unit_vel.shape), len(mask.shape)),
        )
        unit_id = unsqueeze(unit_id, -3)  # (...,1,n,1)
        unit_id = unit_id + unsqueeze(zeros_like(self.acmi_id), -2)  # (...,N,n,1)
        self._all_tgt_id = unit_id  # (...,N,1,1)

        unit_pos = unsqueeze(unit_pos, -3)  # (...,1,n,d)
        unit_vel = unsqueeze(unit_vel, -3)
        ego_pos = unsqueeze(self._pos_e, -2)  # (...,N,1,d)
        ego_vel = unsqueeze(self._vel_e, -2)
        los = unit_pos - ego_pos  # (...,N,n,3)
        dlos = unit_vel - ego_vel
        self._all_los_e = los  # (...,N,n,3)
        self._all_dlos_e = dlos
        self._all_los_mask = unsqueeze(mask.swapaxes(-2, -1), -1)  # (...,N,n,1)

    def _los2info(self):
        """
        由视线计算其他信息
        """
        dtype = self.dtype
        N = self.group_shape[-1]  # 导弹数
        iw_e = unsqueeze(quat_rotate(self._Q_ew, self._E1F), -2)  # (...,N,1,d)
        los_e = self._all_los_e  # (...,N,n,3)
        dij: DoubleNDArr = clip(norm_(los_e, 2, -1, True), 1e-3, None)  # (...,N,n,1)
        cosa: DoubleNDArr = (los_e * iw_e).sum(-1, keepdims=True) / dij  # (...,N,n,1)
        mask = self._all_los_mask  # (...,N,n,1)
        in_ball = (dij < self._det_rmax) & mask  # 在探测球内
        in_detcone = cosa > self._det_cosa  # 在探测锥内 (...,N,n,1)
        in_trkcone = cosa > self._trk_cosa  # 在跟踪锥内
        self._candet = candet = in_detcone & in_ball  # 在大视场内 (...,N,n,1)
        self._cantrk = cantrk = in_trkcone & in_ball  # 在小视场内 (...,N,n,1)
        # self._ndet = self._candet.sum(-2, keepdims=True)  # 大市场的目标数 (...,N,1,1)
        # self._ntrk = self._cantrk.sum(-2, keepdims=True)  # 小视场的目标数 (...,N,1,1)
        anydet: BoolNDArr = self._candet.any(
            -2, keepdims=True
        )  # 大视场有目标 (...,N,1,1)
        anytrk: BoolNDArr = self._cantrk.any(
            -2, keepdims=True
        )  # 小视场有目标 (...,N,1,1)
        if self.DEBUG:
            anydet

        w1: BoolNDArr = where(anytrk, cantrk, candet)  # 视野过滤(>where) (N,n,1)

        all_dlos = self._all_dlos_e  # (N,n,3)
        ener_V: DoubleNDArr = pow(all_dlos, 2).sum(-1, keepdims=True)  # (N,n,1)

        w2 = asarray(
            (w1 * ener_V),
            dtype=dtype,
        )  # (N,n,1)
        w2sum: DoubleNDArr = w2.sum(-2, keepdims=True)  # (N,1,1)
        w2_ = where(anydet, w2 / w2sum, 0)  # (N,n,1)
        all_los = self._all_los_e  # (N,n,3)
        tgt_los = (w2_ * all_los).sum(-2)  # (N,3)
        tgt_dlos = (w2_ * all_dlos).sum(-2)  # (N,3)

        Qew = self._Q_ew
        fake_los = quat_rotate(Qew, self._E1F * self._det_rmax)  # (N,3)
        fake_vel = self._vel_e  # (N,3)

        _anytrk = anytrk.reshape(N, 1)  # (N,1)
        tgt_los = where(_anytrk, tgt_los, fake_los)  # (N,3)
        tgt_dlos = where(_anytrk, tgt_dlos, fake_vel)  # (N,3)
        assert tgt_los.shape == (N, 3), "tgt_los should be (N,3)"
        assert tgt_dlos.shape == (N, 3), "tgt_dlos should be (N,3)"

        self.target_pos_e = tgt_los + self._pos_e  # (N,3)
        self.target_vel_e = tgt_dlos + self._vel_e  # (N,3)

        dij = norm_(tgt_los, dim=-1, keepdim=True)  # (N,1)
        self._update_distance(dij, None)

    def observe(
        self,
        unit_pos: ndarray,
        unit_vel: ndarray,
        unit_id: ndarray,
        mask: ndarray,
        auto_control: bool = True,
    ):
        """
        (引入过滤机制)输入潜在目标信息

        目前只支持同一局部坐标系下的目标搜索
        Args:
            unit_pos (ndarray): 单位位置(NED地轴坐标), shape: (N,3)
            unit_vel (ndarray): 单位速度(NED地轴坐标), shape: (N,3)
            mask (ndarray): 掩码, shape: (N,B), 目标 i 信息对导弹 j 是否可用
            auto_control (bool, optional): 1->解算PNG控制量, 0->不改变控制量. Defaults to True.
        """
        assert len(unit_pos.shape) == 2, "unit_pos should be 2D tensor"
        assert len(unit_vel.shape) == 2, "unit_vel should be 2D tensor"
        assert len(mask.shape) == 2, "mask should be 2D tensor"
        self._update_los(unit_pos, unit_vel, unit_id, mask)
        self._los2info()  # 更新目标信息
        self.set_target(self.target_pos_e, self.target_vel_e, None, None)
        if auto_control:
            nyz = self.png(self.target_pos_e, self.target_vel_e)  # (N,2)
            self.set_action(nyz)  # 设置控制量
        return

    def target_position_e(self, index: SupportedMaskType | None) -> ndarray:
        """感知到的目标NED位置, shape: (N,3)"""
        index = self.proc_to_mask(index)
        return self.target_pos_e[index]

    def target_velocity_e(self, index: SupportedMaskType | None) -> ndarray:
        """感知到的目标速度, shape: (N,3)"""
        index = self.proc_to_mask(index)
        return self.target_vel_e[index]

    def set_target(
        self,
        target_pos_e: ndarray,
        target_vel_e: ndarray,
        target_id: ndarray | None,
        dst_index: SupportedMaskType | None,
        update_distance: bool = True,
    ):
        """
        (简化情形)直接传入目标
        Args:
            target_pos_e (ndarray): 目标位置(NED地轴坐标), shape: (...,N,3)
            target_vel_e (ndarray): 目标速度(NED地轴坐标), shape: (...,N,3)
            target_id (ndarray|None): 目标ID, shape: (...,N,1)
            dst_index (_SupportedIndexType|None): 分配索引, 默认为所有, 长度=N
        """
        msk = self.proc_to_mask(dst_index)
        self.target_pos_e[msk, :] = target_pos_e
        self.target_vel_e[msk, :] = target_vel_e
        if target_id is not None:
            self.target_id[msk, :] = target_id
            if self.DEBUG:
                self.logr.debug(
                    ("target_id<-{}".format(self.target_id.ravel()[[0]].item()),)
                )

        if update_distance:
            los = target_pos_e - self._pos_e[msk]
            dij = norm_(los, dim=-1, keepdim=True)
            self._update_distance(dij, msk)

    def _calc_mass(self, t_s: ndarray, index: SupportedMaskType | None) -> ndarray:
        index = self.proc_to_mask(index)
        return self._m0 - clip(t_s, 0, self._t_thrust_s) * self._dm

    def set_action(
        self,
        value: ndarray | np.ndarray,
    ):
        """
        设置控制量

        Args:
            value (ndarray): 控制量, shape: (N,2)
                ny_cmd 侧向过载, unit: G
                nz_cmd 法向过载系数(NED +Z 为正), unit: G
        """
        logr = self.logr
        if not isinstance(value, ndarray):
            value = ndarray(
                value,
                dtype=self.dtype,
                # device=self.device,
            )
        ny_cmd, nz_cmd = unbind_keepdim(value, -1)
        # 过载限制
        _nyz_max = self._nyz_max  # @set_action
        self._n_w[..., 1:2] = ny_cmd.clip(-_nyz_max, _nyz_max)
        self._n_w[..., 2:3] = nz_cmd.clip(-_nyz_max, _nyz_max)
        if self.DEBUG:
            logr.debug(
                (
                    "ny_d:{:.3g}".format(ny_cmd.ravel()[[0]].item()),
                    "nz_d:{:.3g}".format(nz_cmd.ravel()[[0]].item()),
                )
            )

    def _propagate(self, index: SupportedMaskType | None):
        """(reset&step 复用)运动学关键状态(TAS,Qew,mu)->全体缓存状态"""
        msk = self.proc_to_mask(index)

        # 姿态一致
        self._ppgt_Qew2rpy_ew(msk)

        # 速度一致
        self._ppgt_tas2Vw(msk)
        self._ppgt_Vw2Ve(msk)

        if self._use_geodetic:
            self._ppgt_z2alt(msk)

    def _run_ode(
        self,
        dt_s: ndarray | float,  # 积分步长(不考虑掩码) unit: sec
        t_s: ndarray,  # 初始时间 unit: sec
        action: ndarray,
        pos_e: ndarray,  # 初始位置地轴系坐标
        tas: ndarray,  # 初始真空速
        Qew: ndarray,  # 初始地轴系/体轴系四元数
        mu: ndarray,  # 初始滚转角(必要冗余,处理万向节死锁)
        mass: ndarray,  # 初始质量
        mask: BoolNDArr,
    ):
        r"""求解运动学关键状态转移, 但不修改本体状态\
        控制量约定:\
            ny_cmd 为侧向过载\
            nz_cmd 为法向过载系数(NED +Z 为正)\

        Args:
            ...

        Returns:
            pos_e_next (ndarray): 位置地轴系坐标, shape: (N,3)
            tas_next (ndarray): 真空速, shape: (N,1)
            Qew_next (ndarray): 地轴/风轴四元数, shape: (N,4)
        """
        ode_solver = ode_rk23
        logr = self.logr

        ny, nz = unbind_keepdim(action, -1)  # (...,1)

        f = lambda t, X: self._f(t, X, ny, nz)
        dt_s = dt_s * (self.is_launched() & mask[..., None])
        rst = ode_solver(f, t_s, cat((pos_e, tas, Qew, mu, mass), axis=-1), dt_s)
        pos_e_next, tas_next, Qew_next, mu_next, mass_next = split_(
            rst, [3, 1, 4, 1, 1], -1
        )
        if self.DEBUG:
            logr.debug(
                (
                    # "obj:{}".format(self.__class__.__name__),
                    "id:{}".format(self.acmi_id.ravel()[[0]].item()),
                    "tas:{:.3g}".format(tas.ravel()[[0]].item()),
                    # "|Qew|:{:.3g}".format(Qew_next[0] norm().item()),
                    "pos_e:{}".format(pos_e_next.reshape(-1, 3)[[0]].tolist()),
                    "TAS:{}".format(tas_next.reshape(-1, 1)[[0]].tolist()),
                    "mu:{:.3g}".format(mu_next.ravel()[[0]].item()),
                    "mass:{:.3g}".format(mass_next.ravel()[[0]].item()),
                )
            )
        return pos_e_next, tas_next, Qew_next, mu_next, mass_next

    def _f(
        self,
        t: ndarray,
        X: ndarray,
        ny: ndarray,
        nz: ndarray,
    ):
        """动力学"""
        p_e, tas, Qew, mu, mass = split_(X, [3, 1, 4, 1, 1], -1)

        logr = self.logr
        _0f = self._0F

        mask = t < self._t_thrust_s  # 是否推进 (...,1)
        dot_mass = self._dm * mask  # (...,1)
        T = self._T * mask
        D = self._D(tas, ny, nz)  # (...,1)
        nx = (T - D) / (mass * self._g)  # (...,1)
        n_w = cat([nx, ny, nz], axis=-1)

        g = self._g  # 重力加速度
        a_w = g * n_w  # 过载加速度风轴分量
        if self.use_gravity:
            Qwe = quat_conj(Qew)
            a_w += quat_rotate(Qwe, self.g_e())

        # 旋转角速度
        dot_mu = (-self._dmu_K) * mu
        tas = clip(tas, self._Vmin, self._Vmax)  # 防止过零
        Vinv = 1 / tas
        dot_tas, a_vy, a_vz = unbind_keepdim(a_w, -1)
        _P = dot_mu
        _Q = -a_vz * Vinv
        _R = a_vy * Vinv
        dot_Qew = quat_mul(Qew, 0.5 * cat([_0f, _P, _Q, _R], axis=-1))
        dot_p_e = quat_rotate(Qew, cat([tas, _0f, _0f], axis=-1))  # 惯性速度

        if self.DEBUG:
            pass
            logr.debug(
                (
                    # "obj:{}".format(self.__class__.__name__),
                    # "id:{}".format(self.id[0,].item()),
                    "t: {:.3f}".format(t[0].item()),
                    "nx:{:.3g}".format(nx[0].item()),
                    # "ny:{:.3g}".format(ny[0].item()),
                    # "nz:{:.3g}".format(nz[0].item()),
                    "T:{:.3g}".format(T[0].item()),
                    "D:{:.3g}".format(D[0].item()),
                    # "mu:{:.3g}".format(mu[0].item()),
                    "dmu:{:.3g}".format(dot_mu[0].item()),
                    # "mass:{:.3g}".format(mass[0].item()),
                    "dmass:{:.3g}".format(dot_mass[0].item()),
                )
            )

        dotX = [dot_p_e, dot_tas, dot_Qew, dot_mu, dot_mass]
        dotX = cat(dotX, axis=-1)
        return dotX

    def png(
        self,
        target_position_e: ndarray,  # (N,3)
        target_velocity_e: ndarray,  # (N,3)
    ) -> ndarray:
        """单目标信息->PN体轴法向、侧向过载"""
        # assert target_position_e.shape == (self.batch_size, 3)
        # assert target_velocity_e.shape == (self.batch_size, 3)
        ego_v = self.velocity_e()  # (N,3)
        los = target_position_e - self.position_e()  # LOS (N,3)
        dlos = target_velocity_e - ego_v  # relative velocity (N,3)
        rr: ndarray = (los * los).sum(-1, keepdims=True)
        rr = clip(rr, 1e-6, None)  # 防止除零
        omega = cross(los, dlos, axis=-1) / rr
        ad_e = self._K_PN * cross(
            dlos, omega, axis=-1
        )  # required acceleration in earth frame (...,3)
        ad_b = quat_rotate_inv(self.Q_ew(), ad_e)
        nyz_d = ad_b[..., 1:] / self._g
        if self.use_gravity:
            nyz_d[..., 1:] -= self._1F
        return nyz_d
