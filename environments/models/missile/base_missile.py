import torch
from abc import ABC, abstractmethod
from typing import Literal, TYPE_CHECKING
from collections.abc import Sequence

from ..base_model import BaseModel
if TYPE_CHECKING:
    from environments.models.aircraft import BaseAircraft

class BaseMissile(BaseModel):
    STATUS_LAUNCHED = 0
    STATUS_HIT = 1
    STATUS_MISSED = 2
    def __init__(
        self,
        model_name: str,
        model_color: Literal["Red", "Blue"],
        position_e: torch.Tensor,                   # model initial position in ENU coordinate system, unit: m, shape: [num_envs, num_models, 3]
        target: "BaseAircraft",
        sim_step_size_ms: int = 1,
        device = torch.device("cpu")
    ) -> None:
        model_type = "Missile"
        self.target = target
        super().__init__(model_name, model_color, position_e, sim_step_size_ms, model_type, device)
        
        # simulation parameters
        self.demage = 100.0
        self.exp_radius = 20.0                  # 
        self._t_thrust_s = 3                    # time limitation of engine, unit: s

        # simulation variables
        self._tas = torch.zeros((self.num_envs, 1), device=self.device)
        self.miss_distance = 1e7*torch.zeros((self.num_envs,), device=self.device)
        self.distance_history = torch.full((self.num_envs, 20), 1e7, device=self.device)
    
    @property
    @abstractmethod
    def altitude_m(self) -> torch.Tensor:
        ...

    @property
    def tas(self) -> torch.Tensor:
        """true air speed, unit: m/s, shape: [num_models, ]"""
        return self._tas
    
    @property
    def velocity_a(self) -> torch.Tensor:
        return torch.cat([self._tas, torch.zeros_like(self._tas), torch.zeros_like(self._tas)], dim=-1)
    
    @property
    @abstractmethod
    def velocity_k(self) -> torch.Tensor:
        ...
    
    def reset(self, env_indices: torch.Tensor):
        self._tas[env_indices] = 0.0
        self.miss_distance[env_indices] = 1e7
        self.distance_history[env_indices, :] = 1e7
        super().reset(env_indices)

    def run(self):
        d = torch.norm(self.position_g-self.target.position_g, p=2, dim=-1)
        mask = d < self.miss_distance
        if torch.any(mask):
            indices = torch.where(mask)[0]
            self.miss_distance[indices] = d[indices]
            
        super().run()

    def launch(self, env_indices: Sequence[int] | torch.Tensor | None = None):
        if env_indices is None:
            env_indices = torch.arange(self.num_envs)

        elif isinstance(env_indices, Sequence):
            # check indices
            index_max = max(env_indices)
            index_min = min(env_indices)
            assert index_max < self.num_envs, index_min >= 0
            env_indices = torch.tensor(env_indices)

        elif isinstance(env_indices, torch.Tensor):
            # check indices
            assert len(env_indices.shape) == 1
            index_max = env_indices.max().item()
            index_min = env_indices.min().item()
            assert index_max < self.num_envs, index_min >= 0

        self._status[env_indices] = BaseMissile.STATUS_LAUNCHED
        self._tas[env_indices] = 600.0

    def is_launch(self, env_indices: Sequence[int] | torch.Tensor | None = None) -> torch.Tensor:
        if env_indices is None:
            env_indices = torch.arange(self.num_envs)

        elif isinstance(env_indices, Sequence):
            # check indices
            index_max = max(env_indices)
            index_min = min(env_indices)
            assert index_max < self.num_envs, index_min >= 0
            env_indices = torch.tensor(env_indices)

        elif isinstance(env_indices, torch.Tensor):
            # check indices
            assert len(env_indices.shape) == 1
            index_min = env_indices.min().item()
            assert index_max < self.num_envs, index_min >= 0

        return self._status[env_indices] == BaseMissile.STATUS_LAUNCHED
    
    def hit(self):
        # 判定是否击中
        d = torch.norm(self.position_g-self.target.position_g, p=2, dim=-1)
        flag = d < self.exp_radius
        if torch.any(flag):
            indices = torch.where(flag)[0]

            self._status[indices] = BaseMissile.STATUS_HIT
            self.target.health_point[indices] -= self.demage

    def is_hit(self, env_indices: Sequence[int] | torch.Tensor | None = None) -> torch.Tensor:
        if env_indices is None:
            env_indices = torch.arange(self.num_envs)

        elif isinstance(env_indices, Sequence):
            # check indices
            index_max = max(env_indices)
            index_min = min(env_indices)
            assert index_max < self.num_envs, index_min >= 0
            env_indices = torch.tensor(env_indices)

        elif isinstance(env_indices, torch.Tensor):
            # check indices
            assert len(env_indices.shape) == 1
            index_min = env_indices.min().item()
            assert index_max < self.num_envs, index_min >= 0

        return self._status[env_indices] == BaseMissile.STATUS_HIT

    def miss(self):
        flag_1 = 0.001*self.sim_time_ms > self._t_thrust_s
        
        d = torch.norm(self.target.position_g - self.position_g, p=2, dim=-1)
        self.distance_history = torch.roll(self.distance_history, shifts=-1, dims=1) 
        self.distance_history[..., -1] = d
        
        diffs = self.distance_history.diff(dim=-1)
        flag_2  = (diffs > 0).all(dim=1)
        flag = torch.logical_and(flag_1, flag_2)
        
        if torch.any(flag):
            indices = torch.where(flag)[0]
            self._status[indices] = BaseMissile.STATUS_MISSED

    def is_missed(self, env_indices: Sequence[int] | torch.Tensor | None = None) -> torch.Tensor:
        if env_indices is None:
            env_indices = torch.arange(self.num_envs)

        elif isinstance(env_indices, Sequence):
            # check indices
            index_max = max(env_indices)
            index_min = min(env_indices)
            assert index_max < self.num_envs, index_min >= 0
            env_indices = torch.tensor(env_indices)

        elif isinstance(env_indices, torch.Tensor):
            # check indices
            assert len(env_indices.shape) == 1
            index_max = env_indices.max().item()
            index_min = env_indices.min().item()
            assert index_max < self.num_envs, index_min >= 0

        return self._status[env_indices] == BaseMissile.STATUS_MISSED