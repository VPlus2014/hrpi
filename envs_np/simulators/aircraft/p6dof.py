from __future__ import annotations
import math
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ..base_model import SupportedMaskType
from collections.abc import Sequence
from .base_aircraft import BaseModel, BaseAircraft


# from .base_aircraft import BaseMissile
from ...utils.math_np import (
    normalize,
    quat_rotate,
    quat_mul,
    quat_conj,
    ode_rk45,
    ode_rk23,
    ode_euler,
    delta_rad_reg,
    affcmb,
    DoubleNDArr,
    FloatNDArr,
    ndarray,
    where,
    asarray,
    cat,
    zeros,
    abs_,
    clip,
    BoolNDArr,
    split_,
    norm_,
    unbind_keepdim,
)


class P6DOFPlane(BaseAircraft):
    def __init__(
        self,
        nx_max: float = 1.0,
        nx_min: float = -0.5,
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
            tas (NDArr|float): 初始真空速 unit: m/s, shape: (N, 1), default: 0
            rpy_ew (NDArr|float): 初始速度系姿态欧拉角(mu,gamma,chi) unit: rad, shape: (N, 3), default: 0

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
        self._ic_tas = zeros(
            (bsz, 1),
            # device=device,
            dtype=dtype,
        )
        self._ic_rpy_ew = zeros(
            (bsz, 3),
            # device=device,
            dtype=dtype,
        )
        self._ic_pos_e = zeros(
            (bsz, 3),
            # device=device,
            dtype=dtype,
        )

        # 当前控制量
        self._n_w = zeros(
            (bsz, 3),
            # device=device,
            dtype=dtype,
        )
        self._dmu = zeros(
            (bsz, 1),
            # device=device,
            dtype=dtype,
        )

    def set_ic_tas(self, tas: ndarray | float, dst_index: SupportedMaskType | None):
        """设置初始真空速"""
        dst_index = self.proc_to_mask(dst_index)
        self._ic_tas[dst_index, :] = tas

    def set_ic_rpy_ew(
        self, rpy_ew: ndarray | float, dst_index: SupportedMaskType | None
    ):
        """设置初始速度系姿态欧拉角"""
        dst_index = self.proc_to_mask(dst_index)
        self._ic_rpy_ew[dst_index, :] = rpy_ew

    def set_ic_pos_e(
        self, position_e: ndarray | float, dst_index: SupportedMaskType | None
    ):
        """设置初始位置地轴系坐标"""
        dst_index = self.proc_to_mask(dst_index)
        self._ic_pos_e[dst_index, :] = position_e

    def reset(
        self,
        mask: SupportedMaskType | None,
    ):
        mask = self.proc_to_mask(mask)

        super().reset(mask)
        self._rpy_ew[mask, :] = self._ic_rpy_ew[mask, :]
        self._tas[mask, :] = self._ic_tas[mask, :]
        self._pos_e[mask, :] = self._ic_pos_e[mask, :]

        self._ppgt_rpy_ew2Qew(mask)
        self._propagate(mask)

    def set_action(
        self, action: DoubleNDArr | FloatNDArr, mask: SupportedMaskType | None = None
    ):
        """
        Args:
            action (NDArr|NDArr): 控制量, shape: (N,4), 分别为:\
            - nx_cmd: 期望切向过载指令(NED +X), unit:G.\
            - ny_cmd: 期望横向过载指令(NED +Y), unit:G.\
            - nz_cmd: 期望法向过载指令(NED +Z), unit:G.\
            - dmu_cmd: 期望滚转角速度指令, unit: rad/s.
        """
        mask = self.proc_to_mask(mask)
        logr = self.logr
        device = self.device
        dtype = self.dtype
        if not isinstance(action, ndarray):
            action = asarray(
                action,
                # device=device,
                dtype=dtype,
            )

        nx_d, ny_d, nz_d, dmu_d = unbind_keepdim(action, -1)  # (...,1)

        self._n_w[mask, 0:1] = nx_d
        self._n_w[mask, 1:2] = ny_d
        self._n_w[mask, 2:3] = nz_d
        self._dmu[mask, :] = dmu_d
        if self.DEBUG:
            logr.debug(
                (
                    "id:{:X}".format(self.acmi_id.reshape(-1)[[0]].item()),
                    "nx:{:.3g}".format(nx_d.ravel()[[0]].item()),
                    "ny:{:.3g}".format(ny_d.ravel()[[0]].item()),
                    "nz:{:.3g}".format(nz_d.ravel()[[0]].item()),
                    "dot_mu:{:.3g}".format(dmu_d.ravel()[[0]].item()),
                )
            )

    def action_n2c(self, action: ndarray, linear=True):
        """
        (面向神经网络控制) [-1,1]归一化动作转为底层控制指令, 默认映射方式默认为仿射

        Args:
            action (_type_):
            - nx_cmd: 期望切向过载指令.\
            - ny_cmd: 期望横向过载指令.\
            - nz_cmd: 期望法向过载指令(NED +Z 为正).\
            - dmu_cmd: 期望滚转角速度指令.
        Returns:
            action_c (NDArr): 底层控制指令, shape: (N,4), 分别为:\
            - nx_d: 期望切向过载指令(NED +X), unit:G.\
            - ny_d: 期望横向过载指令(NED +Y), unit:G.\
            - nz_d: 期望法向过载指令(NED +Z), unit:G.\
            - dot_mu: 期望滚转角速度指令, unit: rad/s.
        """
        logr = self.logr
        assert action.shape[-1] == 4, (
            "action must have shape (...,4), but got {}.".format(action.shape),
        )
        nx_cmd, ny_cmd, nz_cmd, dmu_cmd = unbind_keepdim(action, -1)  # (...,1)

        if linear:
            nx_d = affcmb(self._nx_min, self._nx_max, (nx_cmd + 1) * 0.5)
            nz_d = affcmb(-self._nz_up_max, self._nz_down_max, (nz_cmd + 1) * 0.5)
        else:
            nx_d = where(nx_cmd < 0, nx_cmd * -self._nx_min, nx_cmd * self._nx_max)
            nz_d = where(
                nz_cmd < 0, nz_cmd * -self._nz_up_max, nz_cmd * self._nz_down_max
            )  # 期望法向过载
        ny_d = ny_cmd * self._ny_max  # 期望侧向过载
        dot_mu = dmu_cmd * self._dot_mu_max  # 期望滚转角速度
        if self.DEBUG:
            outrange = (action < -1) | (action > 1)
            if outrange.any():
                logr.warning(
                    (
                        "action out of range [-1,1].",
                        where(outrange),
                    )
                )
            logr.debug(
                (
                    "id:{:X}".format(self.acmi_id.ravel()[[0]].item()),
                    "nx_cmd:{:.3g}".format(nx_cmd.ravel()[[0]].item()),
                    "ny_cmd:{:.3g}".format(ny_cmd.ravel()[[0]].item()),
                    "nz_cmd:{:.3g}".format(nz_cmd.ravel()[[0]].item()),
                    "dmu_cmd:{:.3g}".format(dmu_cmd.ravel()[[0]].item()),
                    "nx:{:.3g}".format(nx_d.ravel()[[0]].item()),
                    "ny:{:.3g}".format(ny_d.ravel()[[0]].item()),
                    "nz:{:.3g}".format(nz_d.ravel()[[0]].item()),
                    "dot_mu:{:.3g}".format(dot_mu.ravel()[[0]].item()),
                )
            )
        return cat([nx_d, ny_d, nz_d, dot_mu], axis=-1)

    def run(self, mask: SupportedMaskType | None):
        mask = self.proc_to_mask(mask)
        logr = self.logr
        t = self.sim_time_s()
        pos_e_next, tas_next, Qew_next, mu_next = self._run_ode(
            dt_s=self.sim_step_size_s,
            t_s=t,
            pos_e=self._pos_e,
            tas=self._tas,
            Qew=self._Q_ew,
            mu=self.mu(),
            mask=mask,
        )
        # 后处理
        Qew_next = normalize(Qew_next)
        tas_next = clip(tas_next, self._Vmin, self._Vmax)  # 防止过零

        self._pos_e[mask, :] = pos_e_next[mask, :]
        self._tas[mask, :] = tas_next[mask, :]
        self._Q_ew[mask, :] = Qew_next[mask, :]
        self.set_mu(mu_next[mask, :], mask)

        super().run(mask)  # time++

        self._propagate(mask)
        if self.DEBUG:
            logr.debug(
                {
                    "tas": tas_next.ravel()[[0]].item(),
                    "|Qew|": norm_(Qew_next.reshape(-1, 4)[0, :]).item(),
                    "mu": mu_next.ravel()[[0]].item(),
                }
            )

    def _propagate(self, index: SupportedMaskType | None):
        """(reset&step 复用)运动学关键状态(TAS,Qew,mu)->全体缓存状态"""
        index = self.proc_to_mask(index)
        # 姿态一致
        self._ppgt_Qew2rpy_ew(index)

        # 速度一致
        self._ppgt_tas2Vw(index)
        self._ppgt_Vw2Ve(index)

        if self._use_geodetic:
            self._ppgt_z2alt(index)

    def _run_ode(
        self,
        dt_s: float | ndarray,  # 积分步长(不考虑掩码) unit: sec
        t_s: ndarray,  # 初始时间 unit: sec
        pos_e: ndarray,  # 初始位置地轴系坐标
        tas: ndarray,  # 初始真空速
        Qew: ndarray,  # 初始地轴系/体轴系四元数
        mu: ndarray,  # 初始滚转角(必要冗余,处理万向节死锁)
        mask: BoolNDArr,
    ):
        r"""
        求解运动学关键状态转移, 但不修改本体状态\

        Args:
            ...

        Returns:
            pos_e_next (NDArr): 位置地轴系坐标, shape: (...,N,3)
            tas_next (NDArr): 真空速, shape: (...,N,1)
            Qew_next (NDArr): 地轴/风轴四元数, shape: (...,N,4)
            mu_next (NDArr): 地轴/风轴滚转角, shape: (...,N,1)
        """
        dt_s = dt_s * (self.is_alive() & mask[..., None])
        rst = ode_rk23(self._f, t_s, cat((pos_e, tas, Qew, mu), axis=-1), dt_s)
        pos_e_next, tas_next, Qew_next, mu_next = split_(rst, [3, 1, 4, 1], -1)
        return pos_e_next, tas_next, Qew_next, mu_next

    def _f(self, t: ndarray, X: ndarray):
        """动力学"""
        p_e, V, Qew, mu = split_(X, [3, 1, 4, 1], axis=-1)

        _0 = self._0F
        n_w = self._n_w  # @ode
        dmu = self._dmu  # @ode
        g = self._g

        a_w = g * n_w  # 过载加速度风轴分量
        if self.use_gravity:  # 考虑重力
            Qwe = quat_conj(Qew)
            a_w += quat_rotate(Qwe, self.g_e())

        # 旋转角速度
        V = clip(V, self._Vmin, self._Vmax)  # 防止过零
        dot_V, a_wy, a_wz = unbind_keepdim(a_w, axis=-1)
        Vinv = 1 / V
        _P = dmu
        _Q = -a_wz * Vinv
        _R = a_wy * Vinv
        dot_Qew = quat_mul(Qew, 0.5 * cat([_0, _P, _Q, _R], axis=-1))
        dot_p_e = quat_rotate(Qew, cat([V, _0, _0], axis=-1))  # 惯性速度

        dotX = [dot_p_e, dot_V, dot_Qew, dmu]
        dotX = cat(dotX, axis=-1)
        return dotX
