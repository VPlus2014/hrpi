"""
RSSM (Recurrent State Space Model) - 循环状态空间模型
Dreamerv3的核心组件，用于建模环境的潜在动态
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class RSSMState:
    """RSSM状态，包含隐状态和后验隐状态"""

    h: Tensor  # 隐状态 (hidden state), shape: (B, H)
    z: Tensor  # 随机隐状态 (stochastic latent), shape: (B, D)

    def detach(self):
        """分离梯度"""
        return RSSMState(h=self.h.detach(), z=self.z.detach())

    def to(self, device):
        """移动到指定设备"""
        return RSSMState(h=self.h.to(device), z=self.z.to(device))


class RSSM(nn.Module):
    """
    RSSM (Recurrent State Space Model)

    包含:
    - posterior: 从观测中推断后验隐状态分布
    - prior: 基于前一状态预测先验隐状态分布
    - 循环核心 (GRU): 更新隐状态
    """

    def __init__(
        self,
        obs_embed_dim: int,  # 观测嵌入维度
        action_dim: int,  # 动作维度
        hidden_dim: int = 256,  # 隐状态维度
        latent_dim: int = 32,  # 潜在变量维度
        num_layers: int = 1,  # GRU层数
    ):
        super().__init__()
        self.obs_embed_dim = obs_embed_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # 循环核心 (GRU)
        self.gru = nn.GRU(
            input_size=latent_dim + action_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # 先验网络: 从隐状态预测潜在变量分布的均值和方差
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # mean and logstd
        )

        # 后验网络: 从隐状态和观测嵌入预测潜在变量分布
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + obs_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # mean and logstd
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _get_dist(self, mean: Tensor, logstd: Tensor) -> torch.distributions.Normal:
        """从均值和对数标准差创建正态分布"""
        logstd = torch.clamp(logstd, -20, 2)
        return torch.distributions.Normal(mean, logstd.exp())

    def initial_state(self, batch_size: int, device: torch.device) -> RSSMState:
        """初始化RSSM状态"""
        return RSSMState(
            h=torch.zeros(batch_size, self.hidden_dim, device=device),
            z=torch.zeros(batch_size, self.latent_dim, device=device),
        )

    def prior(self, state: RSSMState, action: Tensor) -> Tuple[Tensor, Tensor]:
        """
        从前一状态预测先验分布

        Args:
            state: 当前的RSSM状态
            action: 当前动作

        Returns:
            prior_mean: 先验均值
            prior_logstd: 先验对数标准差
        """
        # 拼接隐状态和动作
        gru_input = torch.cat([state.z, action], dim=-1)
        # GRU更新隐状态
        output, h_n = self.gru(gru_input.unsqueeze(1), state.h.unsqueeze(0))
        h = h_n[-1]  # 取最后一层的隐状态

        # 预测先验分布
        prior_out = self.prior_net(h)
        prior_mean, prior_logstd = prior_out.chunk(2, dim=-1)

        return prior_mean, prior_logstd

    def posterior(
        self,
        state: RSSMState,
        obs_embed: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        从观测推断后验分布

        Args:
            state: 当前的RSSM状态 (h)
            obs_embed: 观测嵌入

        Returns:
            posterior_mean: 后验均值
            posterior_logstd: 后验对数标准差
        """
        # 拼接隐状态和观测嵌入
        posterior_input = torch.cat([state.h, obs_embed], dim=-1)
        posterior_out = self.posterior_net(posterior_input)
        posterior_mean, posterior_logstd = posterior_out.chunk(2, dim=-1)

        return posterior_mean, posterior_logstd

    def observe(
        self,
        state: RSSMState,
        action: Tensor,
        obs_embed: Tensor,
    ) -> RSSMState:
        """
        观察步骤: 从观测中更新状态

        Args:
            state: 上一状态
            action: 执行的动作
            obs_embed: 观测嵌入

        Returns:
            新的RSSM状态
        """
        # 预测先验
        prior_mean, prior_logstd = self.prior(state, action)
        prior_dist = self._get_dist(prior_mean, prior_logstd)
        z_prior = prior_dist.rsample()  # 采样

        # 推断后验
        posterior_mean, posterior_logstd = self.posterior(state, obs_embed)
        posterior_dist = self._get_dist(posterior_mean, posterior_logstd)
        z_posterior = posterior_dist.rsample()  # 采样

        # 更新GRU得到新的隐状态
        gru_input = torch.cat([z_posterior, action], dim=-1)
        output, h_n = self.gru(gru_input.unsqueeze(1), state.h.unsqueeze(0))
        h_new = h_n[-1]

        return RSSMState(h=h_new, z=z_posterior)

    def imagine(
        self,
        state: RSSMState,
        action: Tensor,
    ) -> RSSMState:
        """
        想象步骤: 仅基于先验预测下一步状态（无观测）

        Args:
            state: 上一状态
            action: 执行的动作

        Returns:
            新的RSSM状态
        """
        # 预测先验
        prior_mean, prior_logstd = self.prior(state, action)
        prior_dist = self._get_dist(prior_mean, prior_logstd)
        z_new = prior_dist.rsample()

        # 更新GRU
        gru_input = torch.cat([z_new, action], dim=-1)
        output, h_n = self.gru(gru_input.unsqueeze(1), state.h.unsqueeze(0))
        h_new = h_n[-1]

        return RSSMState(h=h_new, z=z_new)

    def get_representation(self, state: RSSMState) -> Tensor:
        """
        获取状态表示 (拼接隐状态和随机隐变量)

        Args:
            state: RSSM状态

        Returns:
            状态表示向量
        """
        return torch.cat([state.h, state.z], dim=-1)
