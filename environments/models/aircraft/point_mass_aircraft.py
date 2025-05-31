import torch
import numpy as np
from typing import Literal
from collections.abc import Sequence
from .base_aircraft import BaseModel, BaseAircraft

# from .base_aircraft import BaseMissile
from environments.utils.math import (
    quat_rotate_inverse, normalize,
    Qx, Qy, Qz, quat_rotate, quat_mul, 
)
class PointMassAircraft(BaseAircraft):
    def __init__(
        self,
        model_name: str,
        model_color: Literal["Red", "Blue"],
        position_e: torch.Tensor,
        tas: torch.Tensor,
        # carried_missiles: BaseMissile | None = None,
        carried_missiles = None,
        sim_step_size_ms: int = 1,
        device = torch.device("cpu")
    ) -> None:
        super().__init__(model_name, model_color, position_e, tas, carried_missiles, sim_step_size_ms, device)

        # simulation parameters
        self.m = 7500                # aircraft mass, unit: kg
        self._S = 26                 # reference area, unit: m^2
        self._c_L_alpha = 4.01       #
        self._C_L_max = 0.753        # 
        self._c_D_0 = 0.0169         #
        self._k_D = 0.179            #
        self._a_tmax = 7.0           #
        self._T_max = 219755         # max thrust, unit: N
        self._tau_mu = 0.3           # unit: sec
        self._tau_alpha = 0.3        # unit: sec

        # simulation variables
        self.alpha = torch.zeros((self.num_envs, 1), device=self.device)  # 攻角
        
        chi = torch.zeros((self.num_envs, 1), device=self.device)    # 航迹方位角(Course)
        gamma = torch.zeros((self.num_envs, 1), device=self.device)  # 航迹倾斜角
        # self.q_kg = quat_from_euler_zyx(torch.zeros_like(gamma), gamma, chi)
        self.q_kg = quat_mul(Qz(chi), Qy(gamma))

        self.mu = torch.zeros((self.num_envs, 1), device=self.device)     # 航迹滚转角

    def set_q_kg(self, gamma: torch.Tensor, chi: torch.Tensor, env_indices: Sequence[int] | torch.Tensor | None = None):
        if env_indices is None:
            env_indices = torch.arange(self.num_envs, device=self.device)

        elif isinstance(env_indices, Sequence):
            # check indices
            index_max = max(env_indices)
            index_min = min(env_indices)
            assert index_max < self.num_envs, index_min >= 0
            env_indices = torch.tensor(env_indices, device=self.device)

        elif isinstance(env_indices, torch.Tensor):
            # check indices
            assert len(env_indices.shape) == 1
            env_indices = env_indices.to(device=self.device)
            index_max = env_indices.max().item()
            index_min = env_indices.min().item()
            assert index_max < self.num_envs, index_min >= 0

        gamma = gamma.to(device=self.device)
        chi = chi.to(device=self.device)
        self.q_kg[env_indices] = quat_mul(Qz(chi), Qy(gamma))

    @property
    def velocity_g(self) -> torch.Tensor:
        """model velocity in erth coordinate system, unit: m/s, shape: [num_models, 3]"""
        q_ak = Qx(self.mu)
        velocity_k = quat_rotate_inverse(q=q_ak, v=self.velocity_a)
        velocity_g = quat_rotate_inverse(q=self.q_kg, v=velocity_k)
        return velocity_g
    
    @property
    def altitude_m(self) -> torch.Tensor:
        return -1*self._position_g[..., -1:]

    def reset(self, env_indices: Sequence[int] | torch.Tensor | None = None):
        if env_indices is None:
            env_indices = torch.arange(self.num_envs, device=self.device)

        elif isinstance(env_indices, Sequence):
            # check indices
            index_max = max(env_indices)
            index_min = min(env_indices)
            assert index_max < self.num_envs, index_min >= 0
            env_indices = torch.tensor(env_indices, device=self.device)

        elif isinstance(env_indices, torch.Tensor):
            # check indices
            assert len(env_indices.shape) == 1
            env_indices = env_indices.to(device=self.device)
            index_max = env_indices.max().item()
            index_min = env_indices.min().item()
            assert index_max < self.num_envs, index_min >= 0
        
        super().reset(env_indices)
        self.alpha[env_indices] = torch.zeros((env_indices.shape[0], 1), device=self.device)   # 攻角
        
        chi = torch.zeros((env_indices.shape[0], 1), device=self.device)    # 航迹方位角(Course)
        gamma = torch.zeros((env_indices.shape[0], 1), device=self.device)  # 航迹倾斜角
        self.q_kg[env_indices] = quat_mul(Qz(chi), Qy(gamma))

        self.mu[env_indices] = torch.zeros((env_indices.shape[0], 1), device=self.device)     # 航迹滚转角

    def run(self, action: np.ndarray |torch.Tensor):
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(device=self.device)
        
        self._position_g, self._tas, q_kg, self.mu, self.alpha = self.ode_solver(
            self.position_g, self.velocity_g, self.tas, self.q_kg, self.alpha, self.mu, 
            t_s = 0.001*self.sim_step_size_ms, action = action
        )
        self.q_kg = normalize(q_kg)
        super().run(action)
        
    def ode_solver(
        self, 
        position_g: torch.Tensor, 
        velocity_g: torch.Tensor,
        tas: torch.Tensor, 
        q_kg: torch.Tensor,
        alpha: torch.Tensor,
        mu: torch.Tensor,
        t_s: float, 
        action: torch.Tensor
    ) -> torch.Tensor:
        thrust_cmd, alpha_cmd, mu_cmd = torch.unbind(action, dim=-1)

        q_ak = Qx(mu)
        D_a = self.D_a(tas, alpha, self._rho)
        L_a = self.L_a(tas, alpha, self._rho)
        A_a = torch.cat([-D_a, torch.zeros_like(D_a), -L_a], dim=-1)
        A_k = quat_rotate_inverse(q_ak, A_a)

        T_b = torch.stack([self._T_max*thrust_cmd, torch.zeros_like(thrust_cmd), torch.zeros_like(thrust_cmd)], dim=-1)
        q_ab = Qy(-alpha)
        T_a = quat_rotate(q=q_ab, v=T_b)
        T_k = quat_rotate_inverse(q=q_ak, v=T_a)
        
        acc_k = (A_k + T_k) / self.m + quat_rotate(q_kg, self.G_e)

        dot_tas = acc_k[..., :1]
        dot_alpha = (alpha_cmd.unsqueeze(-1) - alpha)/self._tau_alpha
        dot_mu = (mu_cmd.unsqueeze(-1) - mu)/self._tau_mu
        omega_k = (acc_k - torch.cat([dot_tas, torch.zeros_like(dot_tas), torch.zeros_like(dot_tas)], dim=-1))/((tas+1e-6)*torch.tensor([[1, 1, -1]], device=self.device))
        omega_k = omega_k[:, [0, 2, 1]]
        dot_q_kg = 0.5*quat_mul(q_kg, torch.cat([torch.zeros(size=(omega_k.shape[0], 1), device=self.device), omega_k], dim=-1))

        position_g_prime = position_g + t_s*velocity_g*self.is_alive()
        tas_prime = tas + t_s*dot_tas*self.is_alive()
        q_kg_prime = q_kg + t_s*dot_q_kg*self.is_alive()
        
        mu_prime = mu + t_s*dot_mu*self.is_alive()
        alpha_prime = alpha + t_s*dot_alpha*self.is_alive()

        return position_g_prime, tas_prime, q_kg_prime, mu_prime, alpha_prime
    
    @property
    def G_e(self) -> torch.Tensor:
        return self._g*torch.cat([torch.zeros(size=(self.num_envs, 2), device=self.device), torch.ones(size=(self.num_envs, 1), device=self.device)], dim=-1)

    # 空气动力计算
    def c_L(self, alpha: torch.Tensor) -> torch.Tensor:
        c_L = self._c_L_alpha*alpha
        return c_L
    
    def L_a(self, tas: torch.Tensor, alpha: torch.Tensor, rho: float) -> torch.Tensor:
        L_a = 0.5*rho*tas.pow(2)*self._S*self.c_L(alpha)
        return L_a
    
    def c_D(self, alpha: torch.Tensor) -> torch.Tensor:
        return self._c_D_0 + self._k_D*self.c_L(alpha).pow(2)

    def D_a(self, tas: torch.Tensor, alpha: torch.Tensor, rho: float) -> torch.Tensor:
        D_a = 0.5*rho*tas.pow(2)*self._S*self.c_D(alpha)
        return D_a
    
