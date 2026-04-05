from __future__ import annotations

import numpy as np
from .proto4rf import BaseRewardFn


class TrajectoryReward(BaseRewardFn):
    def forward(self, env, plane):
        """
        根据给定的环境和飞机信息，计算前向轨迹的得分。

        Args:
        env (Environment): 环境信息，包含战略目标和障碍物等信息。
        plane (Aircraft): 飞机信息，包含当前飞机状态、轨迹等信息。

        Returns:
        float: 前向轨迹的得分，由轨迹平滑性得分和决策层目标符合度得分组成。

        """
        # 轨迹平滑性 + 决策层目标符合度
        jerk_penalty = -np.sum(np.abs(plane.jerk)) * 0.1
        goal_alignment = compute_alignment(plane.trajectory, env.strategic_goal)
        return jerk_penalty + goal_alignment
