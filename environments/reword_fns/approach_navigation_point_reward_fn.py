from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..navigation import NavigationEnv
    from ..evasion import EvasionEnv
    from .base_reward_fn import IndexLike

import torch
from .base_reward_fn import BaseRewardFn, IndexLike
from environments.utils.math import euler_from_quat, ned2aer


class ApproachNavigationPointRewardFn(BaseRewardFn):
    def __init__(self, weight: float = 1) -> None:
        super().__init__()
        self.weight = weight

    def reset(self, env: NavigationEnv, env_indices: IndexLike | None = None):
        r = env.navagation_LOS
        try:
            self.pre_distance[env_indices] = torch.norm(r, p=2, dim=-1, keepdim=True)[
                env_indices
            ]
        except AttributeError:
            self.pre_distance = torch.norm(r, p=2, dim=-1, keepdim=True)

    def __call__(self, env: "NavigationEnv", **kwargs) -> torch.Tensor:
        los = env.navagation_LOS

        euler = euler_from_quat(env.aircraft.q_kg)
        gamma = euler[..., 1:2]
        chi = euler[..., 2:3]

        aer = ned2aer(los)
        # print("ApproachNavigationPointRewardFn\n az: {}, chi: {}".format(aer[0, 0:1], chi[0, 0:1]))
        chi_reward = (
            4
            - torch.abs(torch.sin(aer[..., 0:1]) - torch.sin(chi))
            - torch.abs(torch.cos(aer[..., 0:1]) - torch.cos(chi))
        )
        gamma_reward = (
            4
            - torch.abs(torch.sin(aer[..., 1:2]) - torch.sin(gamma))
            - torch.abs(torch.cos(aer[..., 1:2]) - torch.cos(gamma))
        )
        distance_reward = -aer[..., 2:3] / torch.norm(
            (env.position_max_limit - env.position_min_limit).to(dtype=aer.dtype),
            p=2,
        )

        r = selected_points - env.aircraft.position_g
        # v = env.aircraft.velocity_g
        # v_proj = torch.einsum("ij,ij->i", [v, r])/torch.norm(r, p=2, dim=-1)
        # return self.weight*v_proj.unsqueeze(-1)/340
        distance = torch.norm(r, p=2, dim=-1, keepdim=True)
        reward = self.pre_distance - distance
        self.pre_distance = distance
        return self.weight * reward
        # return self.weight*(chi_reward+5*distance_reward)
