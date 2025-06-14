import torch
import numpy as np
from typing import Literal
from collections.abc import Sequence
from .base_missile import BaseModel, BaseMissile
from ..aircraft.base_aircraft import BaseAircraft

# from .base_aircraft import BaseMissile
from environments.utils.math import (
    quat_rotate_inverse,
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
        self._vel_g = torch.zeros(
            (self.batchsize, 3), device=device, dtype=dtype
        )  # ground velocity, unit: m/s, shape: (n,3)
        self._ppgt_vk2vg()

    @property
    def Qgk(self) -> torch.Tensor:
        """从航迹系到地轴系的四元数"""
        return self._Qgk

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
        self.Qgk[env_indices] = quat_mul(Qz(chi), Qy(gamma))

    @property
    def m(self) -> torch.Tensor:
        return (
            self._m0
            - torch.clamp(0.001 * self.sim_time_ms, max=self._t_thrust_s).unsqueeze(-1)
            * self._dm
        )

    @property
    def altitude_m(self) -> torch.Tensor:
        return -1 * self.position_g[..., -1:]

    @property
    def velocity_k(self) -> torch.Tensor:
        return self.velocity_a

    @property
    def velocity_g(self) -> torch.Tensor:
        return self._vel_g

    def reset(self, env_indices: Sequence[int] | torch.Tensor | None = None):
        if env_indices is None:
            env_indices = torch.arange(self.batchsize, device=self.device)

        elif isinstance(env_indices, Sequence):
            # check indices
            index_max = max(env_indices)
            index_min = min(env_indices)
            assert index_max < self.batchsize, index_min >= 0
            env_indices = torch.tensor(env_indices, device=self.device)

        elif isinstance(env_indices, torch.Tensor):
            # check indices
            assert len(env_indices.shape) == 1
            env_indices = env_indices.to(device=self.device)
            index_max = env_indices.max().item()
            index_min = env_indices.min().item()
            assert index_max < self.batchsize, index_min >= 0

        super().reset(env_indices)

        # reset simulation variaode_solverbles
        chi = torch.zeros(
            (env_indices.shape[0], 1), device=self.device
        )  # 航迹方位角(Course)
        gamma = torch.zeros((env_indices.shape[0], 1), device=self.device)  # 航迹倾斜角
        self.Qgk[env_indices] = quat_mul(Qz(chi), Qy(gamma))

    def run(self, action: torch.Tensor):
        (
            self.position_g,
            self.tas,
            q_kg,
        ) = self.ode_solver(
            self.position_g,
            self.velocity_g,
            self.tas,
            self.Qgk,
            t_s=0.001 * self.sim_step_size_ms,
            action=action,
        )
        #
        self._Qgk.copy_(normalize(q_kg))
        self._ppgt_vk2vg()
        self._ppgt_tas2va()

        super().run()

        self.step_hit()

    def _ppgt_vk2vg(self):
        self._vel_g.copy_(quat_rotate(q=self.Qgk, v=self.velocity_k))

    def ode_solver(
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
        mask = 0.001 * self.sim_time_ms < self._t_thrust_s
        T_b = torch.tensor([self._T], device=self.device).repeat(self.batchsize)
        return (mask * T_b).unsqueeze(-1)

    def D_w(
        self, tas: torch.Tensor, n_y: torch.Tensor, n_z: torch.Tensor
    ) -> torch.Tensor:
        D_1 = self._k_1 * torch.pow(tas, 2)
        D_2 = self._K_2 * (torch.pow(n_y, 2) + torch.pow(n_z, 2)) / torch.pow(tas, 2)

        return D_1 + D_2
