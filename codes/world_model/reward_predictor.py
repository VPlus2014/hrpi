"""
奖励预测器 (Reward Predictor)
从RSSM状态预测奖励
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Sequence


class RewardPredictor(nn.Module):
    """
    奖励预测器

    从RSSM状态表示预测即时奖励
    """

    def __init__(
        self,
        state_dim: int,                # RSSM状态维度 (hidden + latent)
        hidden_dims: Sequence[int] = (256, 256), # 隐藏层维度
        output_dim: int = 1,           # 输出维度 (标量奖励)
        scale: float = 1.0,            # 奖励缩放因子
    ):
        super().__init__()
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.scale = scale

        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.SiLU(),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, state: Tensor) -> Tensor:
        """
        预测奖励

        Args:
            state: RSSM状态表示, shape: (B, state_dim) 或 (T, B, state_dim)

        Returns:
            预测的奖励, shape: (B,) 或 (T, B)
        """
        if state.dim() == 3:
            # 序列输入: (T, B, state_dim)
            state_flat = state.flatten(0, 1)
            reward = self.net(state_flat).squeeze(-1)
            T, B = state.shape[:2]
            reward = reward.view(T, B)
        else:
            # 批量输入: (B, state_dim)
            reward = self.net(state).squeeze(-1)

        return reward * self.scale

    def log_prob(
        self,
        state: Tensor,
        target_reward: Tensor,
    ) -> Tensor:
        """
        计算预测奖励的对数概率（用于训练）

        Args:
            state: RSSM状态表示
            target_reward: 目标奖励

        Returns:
            对数概率
        """
        pred_reward = self.forward(state)
        # 假设高斯分布，方差为1
        log_prob = -0.5 * (pred_reward - target_reward) ** 2
        return log_prob
