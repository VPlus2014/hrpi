from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np


# import torch
from .proto4rf import BaseRewardFn, RewardType

if TYPE_CHECKING:
    from ..navigation import NavigationEnv
    from ..nav_heading import NavHeadingEnv
    from ._rt_head import EnvMaskType, SyncVecEnv


class RF_GoalDistance(BaseRewardFn):
    """距离代价"""
    def __init__(
        self,
        use_dR: bool = False,  # 1:路程, 0:平均距离
        use_quadratic: bool = False,
        use_history: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._params.update(
            {
                "use_dR": use_dR,
                "use_quadratic": use_quadratic,
                "use_history": use_history,
            }
        )
        self.use_dR = use_dR
        self.use_quadratic = use_quadratic
        self.use_history = use_history

    def reset(self, env: NavHeadingEnv, env_mask, **kwargs):
        if self.use_dR:
            if self.use_history:
                msk = env.proc_to_mask(env_mask)
                try:
                    self._last_los[msk, :] = env.goal_LOS
                except AttributeError:
                    self._last_los = env.goal_LOS.copy()
            else:
                self._dt = env._agent_step_size_ms * 1e-3

    def forward(self, env: NavHeadingEnv, unit, **kwargs) -> RewardType:
        from ._rt_head import math_np

        if self.use_dR:
            if self.use_history:
                los = env.goal_LOS
                dp = los - self._last_los
                self._last_los[...] = los
            else:
                dp = unit.velocity_e() * self._dt
            cost = math_np.norm_(dp)
        else:
            cost = env.goal_distance

        if self.use_quadratic:
            cost = cost**2

        reward = -cost
        return reward
        # return self.weight*(chi_reward+5*distance_reward)
