import torch
from typing import TYPE_CHECKING, Sequence
from ..base_reward_fn import BaseRewardFn
from environments.utils.math import euler_from_quat, ned2aer
if TYPE_CHECKING:
    from environments.evasion import EvasionEnv


class ApproachNavigationPointRewardFn(BaseRewardFn):
    def __init__(self, weight: float):
        super().__init__()
        self.weight = weight
    
    def reset(self, env: "EvasionEnv", env_indices: torch.Tensor):
        pass
        
    def __call__(self, env: "EvasionEnv", **kwargs) -> torch.Tensor:
        r = env.missile.target.position_g-env.missile.position_g
        v = env.missile.velocity_g
        v_proj = torch.einsum("ij,ij->i", [v, r])/torch.norm(r, p=2, dim=-1)
        return -1*self.weight*v_proj.unsqueeze(-1)