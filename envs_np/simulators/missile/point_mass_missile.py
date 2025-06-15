from __future__ import annotations
from typing import TYPE_CHECKING

# import torch
import numpy as np
from collections.abc import Sequence
from .base_missile import BaseMissile, BaseModel, SupportedMaskType
from ..aircraft.base_aircraft import BaseAircraft

# from .base_aircraft import BaseMissile
from environments_th.utils.math_pt import (
    quat_rotate_inv,
    normalize,
    Qx,
    Qy,
    Qz,
    quat_rotate,
    quat_mul,
    rpy2quat,
)


class PointMassMissile(BaseMissile):
    pass
    # def __init__(
    #     self,
    #     target: BaseModel,
    #     kill_radius=10.0,
    #     **kwargs,
    # ) -> None:
    #     super().__init__(target=target, kill_radius=kill_radius, **kwargs)
    #     device = self.device
    #     dtype = self.dtype

    #     # simulation parameters
    #     self._m0 = 84  # initial mass, unit: kg
    #     self._dm = 6.0  # mass loss rate, unit: kg/s
    #     self._T = 7063.2  # thrust, unit: N
    #     self._N = 3  # proportionality constant of proportional navigation
    #     self._nyz_max = 30  # max overload
    #     self._t_thrust_s = 8  # time limitation of engine, unit: s
    #     self._k_1 = 0.001
    #     self._K_2 = 1

    # @property
    # def m(self) -> np.ndarray:
    #     return (
    #         self._m0
    #         - np.clamp(self.sim_time_s(), max=self._t_thrust_s).unsqueeze(-1)
    #         * self._dm
    #     )

    # def reset(self, env_indices: Sequence[int] | np.ndarray | None = None):
    #     env_indices = self.proc_index(env_indices)
    #     super().reset(env_indices)

    #     self._tas[env_indices] = 600.0
    #     self._ppgt_tas2Vw()

    #     # reset simulation variaode_solverbles
    #     chi = np.zeros(
    #         (env_indices.shape[0], 1), device=self.device
    #     )  # 航迹方位角(Course)
    #     gamma = np.zeros((env_indices.shape[0], 1), device=self.device)  # 航迹倾斜角
    #     self.Q_ew(env_indices) = quat_mul(Qz(chi), Qy(gamma))

    # def run(self, action: np.ndarray):
    #     (
    #         self.position_e,
    #         self.tas,
    #         q_kg,
    #     ) = self._run_ode(
    #         self.position_e,
    #         self.velocity_e,
    #         self.tas,
    #         self.Q_ew,
    #         t_s=0.001 * self._sim_step_size_ms,
    #         action=action,
    #     )
    #     #
    #     self._Qgk.copy_(normalize(q_kg))
    #     self._ppgt_Vb2Ve()
    #     self._ppgt_tas2Vw()

    #     super().run()

    # def launch(self, env_indices):
    #     env_indices = self.proc_index(env_indices)
    #     super().launch(env_indices)

    #     self._tas[env_indices] = 600.0
    #     self._ppgt_tas2Vw()

    # def _run_ode(
    #     self,
    #     position_e: np.ndarray,
    #     velocity_e: np.ndarray,
    #     tas: np.ndarray,
    #     q_kg: np.ndarray,
    #     t_s: np.ndarray | float,
    #     action: np.ndarray,
    # ):
    #     n_y, n_z = np.split(action, [1, 1], axis=-1)  # (...,1)
    #     D_w = self.D_w(tas, n_y, n_z)
    #     _0 = np.zeros_like(D_w)
    #     A_w = np.cat([-D_w, _0, _0], axis=-1)
    #     A_k = A_w

    #     T_b = np.cat(
    #         [self.T_b, np.zeros_like(self.T_b), np.zeros_like(self.T_b)], dim=-1
    #     )
    #     T_k = T_b

    #     n = np.cat([np.zeros_like(n_y), n_y, n_z], axis=-1)
    #     acc_k = self._g * n + (A_k + T_k) / self.m + quat_rotate(q_kg, self.G_e)

    #     dot_tas = acc_k[..., :1]

    #     omega_k = (
    #         acc_k
    #         - np.cat(
    #             [dot_tas, np.zeros_like(dot_tas), np.zeros_like(dot_tas)], dim=-1
    #         )
    #     ) / (tas.clip(min=1e-3) * np.tensor([[1, 1, -1]], device=self.device))
    #     omega_k = omega_k[:, [0, 2, 1]]
    #     dot_q_kg = 0.5 * quat_mul(
    #         q_kg,
    #         np.cat(
    #             [np.zeros(size=(omega_k.shape[0], 1), device=self.device), omega_k],
    #             dim=-1,
    #         ),
    #     )

    #     alive = self.is_launch()
    #     position_e_prime = position_e + t_s * (velocity_e * alive)
    #     tas_prime = tas + t_s * (dot_tas * alive)
    #     q_ke_prime = q_kg + t_s * (dot_q_kg * alive)

    #     return position_e_prime, tas_prime, q_ke_prime

    # @property
    # def G_e(self) -> np.ndarray:
    #     return self._g_e

    # def _calc_mass(self, t_s: np.ndarray):
    #     """计算质量, shape=(batch_size, 1)"""
    #     m = self._m0 - self._dm * np.clamp(t_s, 0, self._t_thrust_s)
    #     return m

    # @property
    # def T_b(self) -> np.ndarray:
    #     mask = self.sim_time_s < self._t_thrust_s
    #     T_b = np.tensor([self._T], device=self.device).repeat(self.group_size)
    #     return (mask * T_b).unsqueeze(-1)

    # def D_w(
    #     self, tas: np.ndarray, n_y: np.ndarray, n_z: np.ndarray
    # ) -> np.ndarray:
    #     D_1 = self._k_1 * np.pow(tas, 2)
    #     D_2 = self._K_2 * (np.pow(n_y, 2) + np.pow(n_z, 2)) / np.pow(tas, 2)

    #     return D_1 + D_2
