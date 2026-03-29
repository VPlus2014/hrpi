"""
继续预测器 (Continue Predictor / Terminal Predictor)
预测episode是否继续（用于生成想象轨迹）
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Sequence


class ContinuePredictor(nn.Module):
    """
    继续预测器

    从RSSM状态预测episode是否继续
    输出一个在[0,1]范围内的概率值，表示继续的概率
    """

    def __init__(
        self,
        state_dim: int,                # RSSM状态维度 (hidden + latent)
        hidden_dims: Sequence[int] = (256, 256), # 隐藏层维度
        scale: float = 1.0,            # 缩放因子
    ):
        super().__init__()
        self.state_dim = state_dim
        self.scale = scale

        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.SiLU(),
            ])
            prev_dim = hidden_dim

        # 使用sigmoid输出[0,1]范围内的概率
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

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
        预测继续概率

        Args:
            state: RSSM状态表示, shape: (B, state_dim) 或 (T, B, state_dim)

        Returns:
            继续概率, shape: (B,) 或 (T, B)
        """
        if state.dim() == 3:
            # 序列输入: (T, B, state_dim)
            state_flat = state.flatten(0, 1)
            cont = self.net(state_flat).squeeze(-1)
            T, B = state.shape[:2]
            cont = cont.view(T, B)
        else:
            # 批量输入: (B, state_dim)
            cont = self.net(state).squeeze(-1)

        return cont * self.scale

    def log_prob(
        self,
        state: Tensor,
        target_continue: Tensor,
    ) -> Tensor:
        """
        计算继续预测的对数概率

        Args:
            state: RSSM状态表示
            target_continue: 目标继续标志 (0或1)

        Returns:
            对数概率
        """
        pred_continue = self.forward(state)
        # 伯努利分布的对数概率
        # 避免log(0)
        pred_continue = torch.clamp(pred_continue, min=1e-6, max=1-1e-6)
        log_prob = target_continue * torch.log(pred_continue) + \
                   (1 - target_continue) * torch.log(1 - pred_continue)
        return log_prob
