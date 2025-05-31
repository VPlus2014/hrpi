import torch
from abc import abstractmethod
from typing import Literal, TYPE_CHECKING
from collections.abc import Sequence
from copy import deepcopy

from environments.models.base_model import BaseModel

# if TYPE_CHECKING:
#     from ..missile import BaseMissile

class BaseAircraft(BaseModel):
    STATUS_ALIVE = 0
    STATUS_CRASH = 1
    STATUS_SHOTDOWN = 2
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
        model_type = "Aircraft",
        super().__init__(model_name, model_color, position_e, sim_step_size_ms, model_type, device)

        # simulation variables
        self.health_point = torch.tensor([100.0]*self.num_envs, device=self.device)    # health point, shape: [num_models, ]
        self._init_tas = tas.to(device=self.device)
        self._tas = self._init_tas.clone()                              # true air speed, unit: m/s, shape: [num_models, 1]

        self.carried_missiles = carried_missiles
    
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
        """model velocity in velocity coordinate system, unit: m/s, shape: [num_models, 3]"""
        return torch.cat([self._tas, torch.zeros_like(self._tas), torch.zeros_like(self._tas)], dim=-1)
    
    @property
    def velocity_k(self) -> torch.Tensor:
        """model velocity in velocity coordinate system, unit: m/s, shape: [num_models, 3]"""
        return torch.cat([self._tas, torch.zeros_like(self._tas), torch.zeros_like(self._tas)], dim=-1)

    def reset(self, env_indices: Sequence[int] | torch.Tensor | None = None):
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
        
        super().reset(env_indices)
        self.health_point[env_indices] = 100.0
        self._tas[env_indices] = self._init_tas[env_indices]

    def run(self, action: torch.Tensor):
        super().run()

    def activate(self, env_indices: Sequence[int] | torch.Tensor | None = None):
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

        self._status[env_indices] = BaseAircraft.STATUS_ALIVE

    # def launch_missile(self, target_aircraft: "BaseAircraft") -> None:
    #     if self.missiles_num > 0:
    #         missile = self.carried_missiles.pop()
    #         missile.launch(carrier_aircraft=self, target_aircraft=target_aircraft)
    #         self.launched_missiles.append(missile)
    
    def is_alive(self, env_indices: Sequence[int] | torch.Tensor | None = None) -> torch.Tensor:
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

        return self._status[env_indices] == BaseAircraft.STATUS_ALIVE

    def crash(self, env_indices: Sequence[int] | torch.Tensor | None = None):
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

        self._status[env_indices] = BaseAircraft.STATUS_CRASH

    def is_crash(self, env_indices: Sequence[int] | torch.Tensor | None = None) -> torch.Tensor:
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

        return self._status[env_indices] == BaseAircraft.STATUS_CRASH

    def is_shotdown(self, env_indices: Sequence[int] | torch.Tensor | None = None) -> torch.Tensor:
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

        flag = self.health_point <= 1e-6
        if torch.any(flag):
            indices = torch.where(flag)[0]
            
            self._status[indices] = BaseAircraft.STATUS_SHOTDOWN

        return self._status[env_indices] == BaseAircraft.STATUS_SHOTDOWN
        