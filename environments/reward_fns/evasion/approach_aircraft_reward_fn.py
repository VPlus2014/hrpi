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

    def reset(self, env, env_indices=None, **kwargs):
        pass

    def forward(self, env: "EvasionEnv", plane, **kwargs) -> torch.Tensor:
        mis = env.missile
        r = mis.position_e() - plane.position_e()  # 实际的视线
        dr = mis.velocity_e() - plane.velocity_e()  # 实际的相对速度
        Vesc = (r * dr).sum(-1) / torch.norm(r, p=2, dim=-1).clamp(min=1e-3)  # 逃逸速率
        # Vc = torch.einsum("ij,ij->i", [v, r]) / torch.norm(r, p=2, dim=-1)
        return Vesc
