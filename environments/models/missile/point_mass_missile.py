from __future__ import annotations
from typing import TYPE_CHECKING
import torch
import numpy as np
from collections.abc import Sequence
from .base_missile import BaseMissile
from ..aircraft.base_aircraft import BaseAircraft

# from .base_aircraft import BaseMissile
from environments.utils.math import (
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
    def __init__(
        self,
        target: BaseAircraft,
        **kwargs,
    ) -> None:
        super().__init__(target=target, **kwargs)
        device = self.device
        dtype = self.dtype

        # simulation parameters
        self._m0 = 84  # initial mass, unit: kg
        self._dm = 6.0  # mass loss rate, unit: kg/s
        self._T = 7063.2  # thrust, unit: N
        self._N = 3  # proportionality constant of proportional navigation
        self._nyz_max = 30  # max overload
        self._t_thrust_s = 8  # time limitation of engine, unit: s
        self._k_1 = 0.001
        self._K_2 = 1

        # simulation variables
        self._chi = chi = torch.zeros(
            (self.batchsize, 1), device=device, dtype=dtype
        )  # 航迹方位角(Course)
        self._gamma = gamma = torch.zeros(
            (self.batchsize, 1), device=device, dtype=dtype
        )  # 航迹倾斜角
        self._Qgk = quat_mul(Qz(chi), Qy(gamma))
        self._ppgt_vb2ve()

    def set_q_gk(
        self,
        gamma: torch.Tensor,  # (...,1)
        chi: torch.Tensor,  # (...,1)
        env_indices: Sequence[int] | torch.Tensor | None = None,
    ):
        env_indices = self.proc_indices(env_indices)
        device = self.device
        dtype = self.dtype

        gamma = gamma.to(device=device, dtype=dtype)
        chi = chi.to(device=device, dtype=dtype)
        self.Q_ea[env_indices] = quat_mul(Qz(chi), Qy(gamma))

    @property
    def m(self) -> torch.Tensor:
        return (
            self._m0
            - torch.clamp(self.sim_time_s, max=self._t_thrust_s).unsqueeze(-1)
            * self._dm
        )

    @property
    def altitude_m(self) -> torch.Tensor:
        return -1 * self.position_e[..., -1:]

    @property
    def velocity_k(self) -> torch.Tensor:
        return self.velocity_b

    @property
    def velocity_e(self) -> torch.Tensor:
        return self._vel_e

    def reset(self, env_indices: Sequence[int] | torch.Tensor | None = None):
        env_indices = self.proc_indices(env_indices)
        super().reset(env_indices)

        self._tas[env_indices] = 600.0
        self._ppgt_tas2va()

        # reset simulation variaode_solverbles
        chi = torch.zeros(
            (env_indices.shape[0], 1), device=self.device
        )  # 航迹方位角(Course)
        gamma = torch.zeros((env_indices.shape[0], 1), device=self.device)  # 航迹倾斜角
        self.Q_ea[env_indices] = quat_mul(Qz(chi), Qy(gamma))

    def run(self, action: torch.Tensor):
        (
            self.position_e,
            self.tas,
            q_kg,
        ) = self._run_ode(
            self.position_e,
            self.velocity_e,
            self.tas,
            self.Q_ea,
            t_s=0.001 * self._sim_step_size_ms,
            action=action,
        )
        #
        self._Qgk.copy_(normalize(q_kg))
        self._ppgt_vb2ve()
        self._ppgt_tas2va()

        super().run()

        self.step_hit()

    def launch(self, env_indices):
        env_indices = self.proc_indices(env_indices)
        super().launch(env_indices)
        
        self._tas[env_indices] = 600.0
        self._ppgt_tas2va()

    def _run_ode(
        self,
        position_e: torch.Tensor,
        velocity_e: torch.Tensor,
        tas: torch.Tensor,
        q_kg: torch.Tensor,
        t_s: torch.Tensor | float,
        action: torch.Tensor,
    ):
        n_y, n_z = torch.split(action, [1, 1], -1)  # (...,1)
        D_w = self.D_w(tas, n_y, n_z)
        _0 = torch.zeros_like(D_w)
        A_w = torch.cat([-D_w, _0, _0], -1)
        A_k = A_w

        T_b = torch.cat(
            [self.T_b, torch.zeros_like(self.T_b), torch.zeros_like(self.T_b)], dim=-1
        )
        T_k = T_b

        n = torch.cat([torch.zeros_like(n_y), n_y, n_z], dim=-1)
        acc_k = self._g * n + (A_k + T_k) / self.m + quat_rotate(q_kg, self.G_e)

        dot_tas = acc_k[..., :1]

        omega_k = (
            acc_k
            - torch.cat(
                [dot_tas, torch.zeros_like(dot_tas), torch.zeros_like(dot_tas)], dim=-1
            )
        ) / (tas.clip(min=1e-3) * torch.tensor([[1, 1, -1]], device=self.device))
        omega_k = omega_k[:, [0, 2, 1]]
        dot_q_kg = 0.5 * quat_mul(
            q_kg,
            torch.cat(
                [torch.zeros(size=(omega_k.shape[0], 1), device=self.device), omega_k],
                dim=-1,
            ),
        )

        alive = self.is_launch()
        position_e_prime = position_e + t_s * (velocity_e * alive)
        tas_prime = tas + t_s * (dot_tas * alive)
        q_ke_prime = q_kg + t_s * (dot_q_kg * alive)

        return position_e_prime, tas_prime, q_ke_prime

    @property
    def G_e(self) -> torch.Tensor:
        return self._g * torch.cat(
            [
                torch.zeros(size=(self.batchsize, 2), device=self.device),
                torch.ones(size=(self.batchsize, 1), device=self.device),
            ],
            dim=-1,
        )

    @property
    def T_b(self) -> torch.Tensor:
        mask = self.sim_time_s < self._t_thrust_s
        T_b = torch.tensor([self._T], device=self.device).repeat(self.batchsize)
        return (mask * T_b).unsqueeze(-1)

    def D_w(
        self, tas: torch.Tensor, n_y: torch.Tensor, n_z: torch.Tensor
    ) -> torch.Tensor:
        D_1 = self._k_1 * torch.pow(tas, 2)
        D_2 = self._K_2 * (torch.pow(n_y, 2) + torch.pow(n_z, 2)) / torch.pow(tas, 2)

        return D_1 + D_2

