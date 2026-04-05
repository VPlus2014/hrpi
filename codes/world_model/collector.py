"""
世界模型数据收集器
与NPSyncVecEnv集成，收集训练数据
"""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import gymnasium as gym
import torch
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class RolloutStorage:
    """ rollout存储，用于保存收集的经验数据 """
    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    dones_float: list = field(default_factory=list)  # 用于continue predictor

    def add(self, obs, action, reward, done, done_float):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.dones_float.append(done_float)

    def to_tensors(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """转换为PyTorch张量"""
        if len(self.observations) == 0:
            return {}

        # 堆叠为 (T, B, *shape) 的张量
        obs = torch.as_tensor(np.stack(self.observations), device=device).float()
        action = torch.as_tensor(np.stack(self.actions), device=device).float()
        reward = torch.as_tensor(np.stack(self.rewards), device=device).float()
        done = torch.as_tensor(np.stack(self.dones), device=device)
        done_float = torch.as_tensor(np.stack(self.dones_float), device=device).float()

        return {
            "observations": obs,
            "actions": action,
            "rewards": reward,
            "dones": done,
            "dones_float": done_float,
        }

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.dones_float.clear()

    def __len__(self):
        return len(self.observations)


class WorldModelCollector:
    """
    世界模型数据收集器

    与NPSyncVecEnv无缝集成，自动收集训练数据
    """

    def __init__(
        self,
        env: gym.Env,  # NPSyncVecEnv实例
        device: str|torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        normalize_obs: bool = True,
        obs_mean: Optional[np.ndarray] = None,
        obs_std: Optional[np.ndarray] = None,
    ):
        """
        初始化收集器

        Args:
            env: NPSyncVecEnv环境实例
            device: 计算设备
            normalize_obs: 是否归一化观测
            obs_mean: 观测均值（用于归一化）
            obs_std: 观测标准差（用于归一化）
        """
        self.env = env
        self.device = torch.device(device)
        self.normalize_obs = normalize_obs

        # 获取环境信息
        self.obs_shape = env.single_observation_space.shape
        self.action_dim = env.single_action_space.shape[0]
        self.num_envs = env.num_envs

        # 归一化参数
        if obs_mean is None or obs_std is None:
            self.obs_mean = np.zeros(self.obs_shape, dtype=np.float32)
            self.obs_std = np.ones(self.obs_shape, dtype=np.float32)
        else:
            self.obs_mean = obs_mean.astype(np.float32)
            self.obs_std = obs_std.astype(np.float32)

        # 存储
        self.rollout = RolloutStorage()

    def collect_rollout(
        self,
        num_steps: int,
        policy_fn=None,
        initial_state=None,
    ) -> Dict[str, torch.Tensor]:
        """
        收集一个rollout的数据

        Args:
            num_steps: 收集步数
            policy_fn: 策略函数，接受观测返回动作。如果为None，使用随机策略
            initial_state: 初始RSSM状态（可选）

        Returns:
            包含观测、动作、奖励、完成的字典
        """
        # 重置环境
        obs, info = self.env.reset()
        obs = self._normalize_obs(obs)

        done = np.zeros(self.num_envs, dtype=bool)

        for _ in range(num_steps):
            # 获取动作
            if policy_fn is None:
                # 随机动作
                action = self.env.action_space.sample()
            else:
                # 使用策略
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = policy_fn(obs_tensor).cpu().numpy()[0]

            # 执行动作
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            next_obs = self._normalize_obs(next_obs)

            # 计算完成标志
            done_float = terminated | truncated

            # 存储
            self.rollout.add(obs, action, reward, done, done_float.astype(np.float32))

            # 更新
            obs = next_obs
            done = done_float

            # 如果所有环境都完成，重置
            if done.all():
                obs, info = self.env.reset()
                obs = self._normalize_obs(obs)
                done = np.zeros(self.num_envs, dtype=bool)

        # 转换为张量
        return self.rollout.to_tensors(self.device)

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """归一化观测"""
        if self.normalize_obs:
            obs = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        return obs

    def update_normalization(self, obs: np.ndarray, momentum: float = 0.99):
        """
        在线更新归一化参数

        Args:
            obs: 新观测
            momentum: 动量
        """
        if self.normalize_obs:
            batch_mean = obs.mean(axis=0)
            batch_std = obs.std(axis=0) + 1e-8

            self.obs_mean = momentum * self.obs_mean + (1 - momentum) * batch_mean
            self.obs_std = momentum * self.obs_std + (1 - momentum) * batch_std


def create_collector_from_env(env, **kwargs) -> WorldModelCollector:
    """
    从环境创建收集器的便捷函数

    Args:
        env: NPSyncVecEnv实例
        **kwargs: 传递给WorldModelCollector的其他参数

    Returns:
        WorldModelCollector实例
    """
    return WorldModelCollector(env, **kwargs)
