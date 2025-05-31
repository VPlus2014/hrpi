import torch
from typing import TYPE_CHECKING, Sequence
from ..base_reward_fn import BaseRewardFn
from environments.utils.math import euler_from_quat, ned2aer
if TYPE_CHECKING:
    from environments.evasion import EvasionEnv


class AircraftSurvivalRewardFn(BaseRewardFn):
    def __init__(self, weight: float):
        super().__init__()
        self.weight = weight
    
    def reset(self, env: "EvasionEnv", env_indices: torch.Tensor):
        pass
        
    def __call__(self, env: "EvasionEnv", **kwargs) -> torch.Tensor:
        return self.weight*env.missile.is_missed()