from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...navigation import NavigationEnv

# import torch
from ..base_reward_fn import BaseRewardFn
from ...utils.math_np import ned2aer, vec_cosine


class ApproachNavigationPointRewardFn(BaseRewardFn):
    def __init__(self, weight: float = 1, version=4, record_dist=False) -> None:
        super().__init__()
        self.weight = weight
        self._version = version
        self._record_dist = record_dist or version == 2

    def reset(self, env: NavigationEnv, env_indices=None, **kwargs):
        los = env.cur_nav_LOS
        if self._record_dist:
            env_indices = env.proc_indices(env_indices)
            try:
                self.pre_distance[env_indices] = env.cur_nav_dist[env_indices]
            except AttributeError:
                self.pre_distance = env.cur_nav_dist.clone()

        self._Rmax = float(
            torch.norm(
                (env.position_max_limit - env.position_min_limit).to(dtype=los.dtype),
                p=2,
            ).item()
        )
        self._Rmaxinv = 1.0 / self._Rmax

    def forward(self, env: NavigationEnv, plane, **kwargs) -> torch.Tensor:
        _version = self._version
        pln = env.aircraft
        los = env.cur_nav_LOS

        # gamma = pln.gamma
        # chi = pln.chi

        # aer = ned2aer(los)
        # azi = env.cur_nav_LOS_azimuth
        # elev = env.cur_nav_LOS_elevation
        dij = env.cur_nav_dist
        # print("ApproachNavigationPointRewardFn\n az: {}, chi: {}".format(aer[0, 0:1], chi[0, 0:1]))
        # chi_reward = (
        #     4
        #     - torch.abs(torch.sin(az) - torch.sin(chi))
        #     - torch.abs(torch.cos(az) - torch.cos(chi))
        # )
        # gamma_reward = (
        #     4
        #     - torch.abs(torch.sin(elev) - torch.sin(gamma))
        #     - torch.abs(torch.cos(elev) - torch.cos(gamma))
        # )
        # distance_reward = -dij / self._dij_max
        Rmaxinv = self._Rmaxinv
        if _version == 1:
            v = pln.velocity_e
            tas = pln.tas
            dR = -((v * los).sum(-1, keepdim=True) / dij)  # \dot|LOS|=(-v,LOS)/|LOS|
            reward = -dR * Rmaxinv  # 对应积分 |LOS(t_0)|-|LOS(t_f)|
            # return self.weight * v_proj.unsqueeze(-1) / 340
        elif _version == 2:  # 势差奖励
            reward = (
                self.pre_distance - dij
            ) * Rmaxinv  # 对应积分 |LOS(t_0)|-|LOS(t_f)|
        elif _version == 3:  # 二次代价
            reward = 1 - (
                (dij * Rmaxinv) ** 2
            )  # 对应积分 \int_{t_0}^{t_f} |LOS(t)|^2 dt
        elif _version == 4:  # L1代价
            reward = 1 - (dij * Rmaxinv)  # 对应积分 T-\int_{t_0}^{t_f} |LOS(t)| dt
        else:
            raise NotImplementedError
        self.pre_distance = dij
        return reward
        # return self.weight*(chi_reward+5*distance_reward)
