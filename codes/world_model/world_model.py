"""
世界模型 (World Model)
整合RSSM、Encoder、Decoder、RewardPredictor和ContinuePredictor
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from typing import Sequence

from .rssm import RSSM, RSSMState
from .encoder import Encoder
from .decoder import Decoder
from .reward_predictor import RewardPredictor
from .continue_predictor import ContinuePredictor


@dataclass
class WorldModelOutput:
    """世界模型输出"""
    prior_mean: Tensor
    prior_logstd: Tensor
    posterior_mean: Tensor
    posterior_logstd: Tensor
    z: Tensor  # 采样的潜在变量
    h: Tensor  # 隐状态
    obs_recon: Tensor  # 观测重建
    reward_pred: Tensor  # 奖励预测
    continue_pred: Tensor  # 继续预测
    kl_loss: Tensor  # KL散度损失


class WorldModel(nn.Module):
    """
    完整的世界模型

    包含:
    - Encoder: 将观测编码为嵌入
    - RSSM: 建模潜在动态
    - Decoder: 从状态重建观测
    - RewardPredictor: 预测奖励
    - ContinuePredictor: 预测episode继续
    """

    def __init__(
        self,
        obs_shape: Union[int, Sequence[int]],  # 观测形状
        action_dim: int,           # 动作维度
        embed_dim: int = 256,       # 嵌入维度
        hidden_dim: int = 256,       # RSSM隐状态维度
        latent_dim: int = 32,        # 潜在变量维度
        reward_hidden_dims: Sequence[int] = (256, 256),
        use_cnn: bool = False,       # 是否使用CNN
        free_nats: float = 1.0,      # KL散度自由项
        kl_scale: float = 1.0,       # KL散度权重
    ):
        super().__init__()

        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.free_nats = free_nats
        self.kl_scale = kl_scale

        # 计算状态维度
        state_dim = hidden_dim + latent_dim

        # 组件
        self.encoder = Encoder(obs_shape, embed_dim, use_cnn=use_cnn)
        self.rssm = RSSM(embed_dim, action_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(state_dim, obs_shape, embed_dim=embed_dim, use_cnn=use_cnn)
        self.reward_predictor = RewardPredictor(state_dim, reward_hidden_dims)
        self.continue_predictor = ContinuePredictor(state_dim, reward_hidden_dims)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> RSSMState:
        """初始化RSSM状态"""
        return self.rssm.initial_state(batch_size, device)

    def observe(
        self,
        obs: Tensor,
        action: Tensor,
        state: Optional[RSSMState] = None,
    ) -> Tuple[RSSMState, WorldModelOutput]:
        """
        观察步骤: 从观测中学习状态

        Args:
            obs: 观测序列, shape: (T, B, *obs_shape)
            action: 动作序列, shape: (T, B, action_dim)
            state: 初始状态（可选）

        Returns:
            状态序列和输出
        """
        T, B = obs.shape[:2]
        device = obs.device

        # 初始化状态
        if state is None:
            state = self.initial_state(B, device)

        # 编码观测
        obs_embed = self.encoder(obs)  # (T, B, embed_dim)

        # 逐时间步处理
        states = []
        outputs = []

        for t in range(T):
            obs_t = obs_embed[t] if obs_embed.dim() == 3 else obs_embed
            action_t = action[t] if action.dim() == 2 else action

            # 观察更新
            new_state = state  # 使用上一个状态
            new_state = self.rssm.observe(new_state, action_t, obs_t)

            # 计算输出
            state_repr = self.rssm.get_representation(new_state)
            obs_recon = self.decoder(state_repr)
            reward_pred = self.reward_predictor(state_repr)
            continue_pred = self.continue_predictor(state_repr)

            # 计算KL散度
            prior_mean, prior_logstd = self.rssm.prior(state, action_t)
            posterior_mean, posterior_logstd = self.rssm.posterior(state, obs_t)
            kl_loss = self._kl_loss(prior_mean, prior_logstd, posterior_mean, posterior_logstd)

            output = WorldModelOutput(
                prior_mean=prior_mean,
                prior_logstd=prior_logstd,
                posterior_mean=posterior_mean,
                posterior_logstd=posterior_logstd,
                z=new_state.z,
                h=new_state.h,
                obs_recon=obs_recon,
                reward_pred=reward_pred,
                continue_pred=continue_pred,
                kl_loss=kl_loss,
            )

            states.append(new_state)
            outputs.append(output)
            state = new_state

        # 堆叠序列
        states = self._stack_states(states)
        outputs = self._stack_outputs(outputs)

        return states, outputs

    def imagine(
        self,
        state: RSSMState,
        actions: Tensor,
    ) -> Tuple[RSSMState, WorldModelOutput]:
        """
        想象步骤: 仅基于先验预测未来

        Args:
            state: 初始状态
            actions: 动作序列, shape: (T, B, action_dim)

        Returns:
            状态序列和输出
        """
        T, B = actions.shape[:2]
        device = actions.device

        states = []
        outputs = []

        for t in range(T):
            action_t = actions[t]

            # 想象更新
            new_state = self.rssm.imagine(state, action_t)

            # 计算输出
            state_repr = self.rssm.get_representation(new_state)
            obs_recon = self.decoder(state_repr)
            reward_pred = self.reward_predictor(state_repr)
            continue_pred = self.continue_predictor(state_repr)

            # 想象时没有KL损失
            kl_loss = torch.zeros(B, device=device)

            output = WorldModelOutput(
                prior_mean=torch.zeros(B, self.latent_dim, device=device),
                prior_logstd=torch.zeros(B, self.latent_dim, device=device),
                posterior_mean=new_state.z,
                posterior_logstd=torch.zeros(B, self.latent_dim, device=device),
                z=new_state.z,
                h=new_state.h,
                obs_recon=obs_recon,
                reward_pred=reward_pred,
                continue_pred=continue_pred,
                kl_loss=kl_loss,
            )

            states.append(new_state)
            outputs.append(output)
            state = new_state

        states = self._stack_states(states)
        outputs = self._stack_outputs(outputs)

        return states, outputs

    def _kl_loss(
        self,
        prior_mean: Tensor,
        prior_logstd: Tensor,
        posterior_mean: Tensor,
        posterior_logstd: Tensor,
    ) -> Tensor:
        """计算KL散度损失"""
        prior_dist = torch.distributions.Normal(prior_mean, prior_logstd.exp())
        posterior_dist = torch.distributions.Normal(posterior_mean, posterior_logstd.exp())

        kl = torch.distributions.kl.kl_divergence(posterior_dist, prior_dist)
        # 自由项：避免KL过小
        kl = torch.clamp(kl, min=self.free_nats)
        return kl

    def _stack_states(self, states: list[RSSMState]) -> RSSMState:
        """堆叠状态序列"""
        return RSSMState(
            h=torch.stack([s.h for s in states], dim=0),
            z=torch.stack([s.z for s in states], dim=0),
        )

    def _stack_outputs(self, outputs: list[WorldModelOutput]) -> WorldModelOutput:
        """堆叠输出序列"""
        return WorldModelOutput(
            prior_mean=torch.stack([o.prior_mean for o in outputs], dim=0),
            prior_logstd=torch.stack([o.prior_logstd for o in outputs], dim=0),
            posterior_mean=torch.stack([o.posterior_mean for o in outputs], dim=0),
            posterior_logstd=torch.stack([o.posterior_logstd for o in outputs], dim=0),
            z=torch.stack([o.z for o in outputs], dim=0),
            h=torch.stack([o.h for o in outputs], dim=0),
            obs_recon=torch.stack([o.obs_recon for o in outputs], dim=0),
            reward_pred=torch.stack([o.reward_pred for o in outputs], dim=0),
            continue_pred=torch.stack([o.continue_pred for o in outputs], dim=0),
            kl_loss=torch.stack([o.kl_loss for o in outputs], dim=0),
        )

    def compute_loss(
        self,
        obs: Tensor,
        action: Tensor,
        reward: Tensor,
        continue_flag: Tensor,
        state: Optional[RSSMState] = None,
    ) -> Tuple[Tensor, dict]:
        """
        计算世界模型训练损失

        Args:
            obs: 观测序列, shape: (T, B, *obs_shape)
            action: 动作序列, shape: (T, B, action_dim)
            reward: 奖励序列, shape: (T, B)
            continue_flag: 继续标志, shape: (T, B)
            state: 初始状态

        Returns:
            总损失和损失分量字典
        """
        T, B = obs.shape[:2]
        device = obs.device

        # 观察
        states, outputs = self.observe(obs, action, state)

        # 计算重建损失 (观测重建)
        obs_flat = obs.flatten(0, 1).flatten(2) if len(self.obs_shape) > 1 else obs.flatten(0, 1)
        recon_flat = outputs.obs_recon.flatten(0, 1).flatten(2) if outputs.obs_recon.dim() > 2 else outputs.obs_recon
        obs_loss = -self.decoder.log_prob(
            states.h.flatten(0, 1),
            obs_flat,
        ).mean()

        # 奖励预测损失
        reward_target = reward.flatten(0, 1)
        reward_pred_flat = outputs.reward_pred.flatten(0, 1)
        reward_loss = -self.reward_predictor.log_prob(
            states.h.flatten(0, 1),
            reward_target,
        ).mean()

        # 继续预测损失
        continue_target = continue_flag.flatten(0, 1)
        continue_pred_flat = outputs.continue_pred.flatten(0, 1)
        continue_loss = -self.continue_predictor.log_prob(
            states.h.flatten(0, 1),
            continue_target,
        ).mean()

        # KL散度损失
        kl_loss = outputs.kl_loss.mean()

        # 总损失
        total_loss = obs_loss + reward_loss + continue_loss + self.kl_scale * kl_loss

        loss_dict = {
            "total_loss": total_loss.item(),
            "obs_loss": obs_loss.item(),
            "reward_loss": reward_loss.item(),
            "continue_loss": continue_loss.item(),
            "kl_loss": kl_loss.item(),
        }

        return total_loss, loss_dict
