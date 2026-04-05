from __future__ import annotations
from .proto4rf import BaseRewardFn

import numpy as np



class StrategicReward(BaseRewardFn):
    def forward(self, env, plane):
        # 生存奖励 + 态势评估
        survival = env.missile.is_missed() * 10
        distance = -np.linalg.norm(plane.position - env.missile.position) * 0.01
        return survival + distance
