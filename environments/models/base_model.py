import torch
from abc import ABC, abstractmethod
from typing import Literal
from collections.abc import Sequence


class BaseModel(ABC):
    STATUS_INACTIVATE = -1
    def __init__(
        self,
        model_name: str,
        model_color: Literal["Red", "Blue"],
        position_g: torch.Tensor,                                   # model initial position in ENU coordinate system, unit: m, shape: [num_envs, 3]
        sim_step_size_ms: int = 1,
        model_type: Literal["Aircraft", "Missile"] = "Aircraft",
        device = torch.device("cpu")
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.model_color = model_color
        self.sim_step_size_ms = sim_step_size_ms
        self.model_type = model_type
        self.device = device
        
        # simulation variables
        assert len(position_g.shape) == 2 and position_g.shape[-1] == 3
        self._init_position_g = position_g.to(device=self.device)   # model initial position in ENU coordinate system, unit: m, shape: [num_envs, 3]
        self._position_g = self._init_position_g.clone()            # model position in ENU coordinate system, unit: m

        # simulation paramters
        self._g = 9.8           # acceleration of gravity, unit: m/s^2
        self._rho = 1.29                                             # atmosphere density, unit: kp/m^3
        
        self._status = BaseModel.STATUS_INACTIVATE*torch.ones(size=(self.num_envs, 1), dtype=torch.int64, device=self.device) # shape: [num_envs, ]
        self._sim_time_ms = torch.tensor([0]*self.num_envs, dtype=torch.int64, device=self.device)

    @property
    def num_envs(self) -> int:
        return self._init_position_g.shape[0]

    @property
    def position_g(self) -> torch.Tensor:
        """
            model position in ENU coordinate system, unit: m

            args:

            rets:
                
        """
        return self._position_g

    @property
    @abstractmethod
    def velocity_g(self, env_indices: Sequence[int] | torch.Tensor | None = None) -> torch.Tensor:
        """model velocity in ENU coordinate system, unit: m/s"""
        ...
    
    @property
    def sim_time_ms(self, env_indices: Sequence[int] | torch.Tensor | None = None) -> torch.Tensor:
        """model simulation time, unit: ms"""
        if isinstance(env_indices, Sequence):
            # check indices
            index_max = max(env_indices)
            index_min = min(env_indices)
            assert index_max < self.num_envs, index_min >= 0

            return self._sim_time_ms[env_indices]
        
        elif isinstance(env_indices, torch.Tensor):
            # check indices
            assert len(env_indices.shape) == 1
            index_min = env_indices.min().item()
            assert index_max < self.num_envs, index_min >= 0

            return self._sim_time_ms[env_indices]
        
        elif env_indices is None:
            return self._sim_time_ms
    
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

        self._position_g[env_indices] = self._init_position_g[env_indices]
        self._status[env_indices] = BaseModel.STATUS_INACTIVATE
        self._sim_time_ms[env_indices] = 0.0

    def run(self):
        self._sim_time_ms += self.sim_step_size_ms
    
