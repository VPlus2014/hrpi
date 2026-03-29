"""
Dreamer Agent
基于世界模型的 imagination-based 智能体
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Sequence

from .world_model import WorldModel, RSSMState
from .rssm import RSSM


class Actor(nn.Module):
    """演员网络：从状态生成动作"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        action_scale: float = 1.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_scale = action_scale

        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.SiLU(),
                ]
            )
            prev_dim = hidden_dim

        # 动作均值和方差
        layers.append(nn.Linear(prev_dim, action_dim))
        self.mean_net = nn.Sequential(*layers[:-1], nn.Linear(prev_dim, action_dim))
        self.log_std_net = nn.Linear(prev_dim, action_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        state: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        从状态生成动作

        Args:
            state: RSSM状态表示
            deterministic: 是否使用确定性策略

        Returns:
            (动作, 动作对数概率)
        """
        mean = self.mean_net(state)
        log_std = self.log_std_net(state)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            action_raw = dist.rsample()
            action = torch.tanh(action_raw)
            # 修正的log prob (因为用了tanh squash)
            log_prob = dist.log_prob(action_raw).sum(dim=-1)
            log_prob -= (2 * (1e-6 + action_raw.tanh().pow(2) + 1e-6).log()).sum(dim=-1)

        action = action * self.action_scale
        return action, log_prob


class Critic(nn.Module):
    """评论家网络：估计状态价值"""

    def __init__(
        self,
        state_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
    ):
        super().__init__()

        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.SiLU(),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, state: Tensor) -> Tensor:
        """估计状态价值"""
        return self.net(state)


@dataclass
class DreamerOutput:
    """Dreamer输出"""

    action: Tensor
    action_log_prob: Tensor
    value: Tensor


class Dreamer(nn.Module):
    """
    Dreamer智能体

    包含:
    - WorldModel: 世界模型
    - Actor: 策略网络
    - Critic: 价值网络

    使用世界模型进行想象轨迹的展开
    """

    def __init__(
        self,
        world_model: WorldModel,
        action_dim: int,
        actor_hidden_dims: Sequence[int] = (256, 256),
        critic_hidden_dims: Sequence[int] = (256, 256),
        gamma: float = 0.99,  # 折扣因子
        lam: float = 0.95,  # GAE lambda
        horizon: int = 15,  # 想象轨迹长度
        action_scale: float = 1.0,
    ):
        super().__init__()

        self.world_model = world_model
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = lam
        self.horizon = horizon

        state_dim = world_model.hidden_dim + world_model.latent_dim

        # 演员和评论家
        self.actor = Actor(
            state_dim, action_dim, actor_hidden_dims, action_scale=action_scale
        )
        self.critic = Critic(state_dim, critic_hidden_dims)

    def initial_state(self, batch_size: int, device: torch.device) -> RSSMState:
        """初始化状态"""
        return self.world_model.initial_state(batch_size, device)

    def act(
        self,
        obs: Tensor,
        state: RSSMState,
        deterministic: bool = False,
    ) -> Tuple[Tensor, RSSMState, DreamerOutput]:
        """
        从观测中选取动作

        Args:
            obs: 当前观测
            state: 当前RSSM状态
            deterministic: 是否使用确定性策略

        Returns:
            (动作, 新状态, Dreamer输出)
        """
        # 编码观测
        obs_embed = self.world_model.encoder(obs)

        # 观察更新
        new_state = self.world_model.rssm.observe(
            state, torch.zeros_like(self._get_dummy_action(state.device)), obs_embed
        )

        # 获取状态表示
        state_repr = self.world_model.rssm.get_representation(new_state)

        # 选取动作
        action, log_prob = self.actor(state_repr, deterministic)
        value = self.critic(state_repr)

        output = DreamerOutput(
            action=action,
            action_log_prob=log_prob,
            value=value,
        )

        return action, new_state, output

    def imagine(
        self,
        state: RSSMState,
        num_steps: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        想象未来的轨迹

        Args:
            state: 初始状态
            num_steps: 想象步数

        Returns:
            (想象的动作序列, 奖励序列, 价值序列)
        """
        device = state.h.device

        # 收集想象轨迹
        imagined_actions = []
        imagined_rewards = []
        imagined_continues = []
        imagined_states = [state]

        curr_state = state

        for _ in range(num_steps):
            # 从当前状态选取动作
            state_repr = self.world_model.rssm.get_representation(curr_state)
            action, _ = self.actor(state_repr, deterministic=True)

            # 预测奖励和继续
            reward_pred = self.world_model.reward_predictor(state_repr)
            continue_pred = self.world_model.continue_predictor(state_repr)

            # 想象下一步
            next_state = self.world_model.rssm.imagine(curr_state, action)

            imagined_actions.append(action)
            imagined_rewards.append(reward_pred)
            imagined_continues.append(continue_pred)
            imagined_states.append(next_state)

            curr_state = next_state

        # 堆叠
        imagined_actions = torch.stack(imagined_actions, dim=0)  # (T, B, A)
        imagined_rewards = torch.stack(imagined_rewards, dim=0)  # (T, B)
        imagined_continues = torch.stack(imagined_continues, dim=0)  # (T, B)

        return imagined_actions, imagined_rewards, imagined_continues

    def compute_returns(
        self,
        rewards: Tensor,
        continues: Tensor,
        bootstrap_value: Tensor,
    ) -> Tensor:
        """
        计算折扣回报 (使用lambda return)

        Args:
            rewards: 奖励序列, shape: (T, B)
            continues: 继续序列, shape: (T, B)
            bootstrap_value: 起始价值, shape: (B,)

        Returns:
            回报序列
        """
        T, B = rewards.shape
        returns = torch.zeros(T + 1, B, device=rewards.device)
        returns[-1] = bootstrap_value

        for t in reversed(range(T)):
            returns[t] = rewards[t] + self.gamma * continues[t] * returns[t + 1]

        return returns[:-1]  # (T, B)

    def _get_dummy_action(self, device: torch.device) -> Tensor:
        """获取虚拟动作（全零）"""
        return torch.zeros(1, self.action_dim, device=device)

    def update_actor_critic(
        self,
        imagined_rewards: Tensor,
        imagined_continues: Tensor,
        imagined_states: list[RSSMState],
    ) -> dict:
        """
        更新演员和评论家

        Args:
            imagined_rewards: 想象奖励
            imagined_continues: 想象继续
            imagined_states: 想象状态序列

        Returns:
            损失字典
        """
        T, B = imagined_rewards.shape

        # 获取每个状态的价值估计
        values = []
        for state in imagined_states[:-1]:  # 不需要最后一个状态的价值
            state_repr = self.world_model.rssm.get_representation(state)
            v = self.critic(state_repr)
            values.append(v)

        values = torch.stack(values, dim=0).squeeze(-1)  # (T, B)

        # Bootstrap价值 (从最后一个状态)
        last_state_repr = self.world_model.rssm.get_representation(imagined_states[-1])
        bootstrap_value = self.critic(last_state_repr).squeeze(-1)

        # 计算returns
        returns = self.compute_returns(
            imagined_rewards, imagined_continues, bootstrap_value
        )

        # 价值损失 (MSE)
        value_loss = F.mse_loss(values, returns)

        # 策略梯度损失 (使用returns作为目标)
        policy_losses = []
        for t, state in enumerate(imagined_states[:-1]):
            state_repr = self.world_model.rssm.get_representation(state)
            action, log_prob = self.actor(state_repr, deterministic=False)

            advantage = returns[t] - values[t].detach()
            policy_loss = -(log_prob * advantage.detach()).mean()

            policy_losses.append(policy_loss)

        policy_loss = torch.stack(policy_losses).mean()

        # 总损失
        total_loss = policy_loss + value_loss

        loss_dict = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, loss_dict
