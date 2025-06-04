from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar
import torch
import torch.nn.functional as F
import numpy as np
from gymnasium import spaces
from typing import cast
from tianshou.data import Batch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

as_tsr = torch.asarray

if TYPE_CHECKING:
    from torch.utils.tensorboard.writer import SummaryWriter

from agents.modules import GaussianActor, Critic
from agents.agent import Agent
from replay_buffer.replaybuffer import ReplayBuffer
from replay_buffer.protocol import RolloutBatchProtocol

from environments.utils.space import normalize, affcmb


class PPOContinuous(Agent):

    def __init__(
        self,
        name: str,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        batch_size: int,
        mini_batch_size: int,
        lr_a: float,
        actor_hidden_sizes: list[int],
        lr_c: float,
        critic_hidden_sizes: list[int],
        gamma: float,
        gae_lambda: float,
        epsilon: float,
        policy_entropy_coef: float,
        use_grad_clip: bool,
        use_adv_norm: bool,
        repeat: int,
        adam_eps: float = 1e-8,
        num_envs: int = 1,
        env_out_torch=True,
        writer: SummaryWriter | None = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float,
    ):
        super().__init__(
            name,
            observation_space,
            action_space=action_space,
            batch_size=batch_size,
            num_envs=num_envs,
            writer=writer,
            device=device,
            dtype=dtype,
        )
        self.mini_batch_size = mini_batch_size

        self.lr_a = lr_a  # Learning rate of actor
        self.lr_c = lr_c  # Learning rate of critic
        self.gamma = gamma  # Discount factor
        self.gae_lambda = gae_lambda  # GAE parameter
        self.epsilon = epsilon  # PPO clip parameter
        self.repeat = repeat  # PPO parameter
        self.entropy_coef = policy_entropy_coef  # Entropy coefficient
        self.adam_eps = adam_eps
        self.use_grad_clip = use_grad_clip
        self.use_adv_norm = use_adv_norm
        self.env_is_torch = env_out_torch

        # 缓存属性
        actmin = torch.from_numpy(action_space.low).to(
            device=self.device, dtype=self.dtype
        )
        actmax = torch.from_numpy(action_space.high).to(
            device=self.device, dtype=self.dtype
        )
        actspan = actmax - actmin
        self.register_buffer("_buf_act_min", actmin)
        self.register_buffer("_buf_act_max", actmax)
        self.register_buffer("_buf_act_span", actspan)
        self._buf_act_min = self.get_buffer("_buf_act_min")
        self._buf_act_max = self.get_buffer("_buf_act_max")
        self._buf_act_span = self.get_buffer("_buf_act_span")

        obsmin = torch.from_numpy(observation_space.low).to(
            device=self.device, dtype=self.dtype
        )
        obsmax = torch.from_numpy(observation_space.high).to(
            device=self.device, dtype=self.dtype
        )
        obsspan = obsmax - obsmin
        self.register_buffer("_buf_obs_min", obsmin)
        self.register_buffer("_buf_obs_max", obsmax)
        self.register_buffer("_buf_obs_span", obsspan)
        self._buf_obs_min = self.get_buffer("_buf_obs_min")
        self._buf_obs_max = self.get_buffer("_buf_obs_max")
        self._buf_obs_span = self.get_buffer("_buf_obs_span")

        self.actor = GaussianActor(
            state_dim=observation_space.shape[0],
            action_dim=action_space.shape[0],
            action_min=self.action_min,
            action_max=self.action_max,
            hidden_sizes=actor_hidden_sizes,
        ).to(device=self.device, dtype=self.dtype)
        self.critic = Critic(
            state_dim=observation_space.shape[0], hidden_sizes=critic_hidden_sizes
        ).to(device=self.device, dtype=self.dtype)

        self.optimizer_actor = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr_a, eps=self.adam_eps
        )
        self.optimizer_critic = torch.optim.Adam(
            self.critic.parameters(), lr=self.lr_c, eps=self.adam_eps
        )

        self.replay_buffer = ReplayBuffer(
            state_dim=observation_space.shape[0],
            action_dim=action_space.shape[0],
            size=self.batch_size,
            num_envs=self.num_envs,
            compact=True,
            device=self.device,
            float_dtype=self.dtype,
        )

        self.returns = torch.zeros(
            size=(self.num_envs, 1), device=self.device, dtype=self.dtype
        )

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        raise NotImplementedError("当前模块不支持迁移设备")

    @property
    def action_min(self) -> torch.Tensor:
        return self._buf_act_min

    @property
    def action_max(self) -> torch.Tensor:
        return self._buf_act_max

    @property
    def observation_min(self) -> torch.Tensor:
        return self._buf_obs_min

    @property
    def observation_max(self) -> torch.Tensor:
        return self._buf_obs_max

    @torch.no_grad()
    def evaluate(self, state: np.ndarray | torch.Tensor) -> torch.Tensor:
        state = as_tsr(state, device=self.device, dtype=self.dtype)  # (...,dimX)
        rst = self._integ_infer(
            state,
            greedy=True,
            with_log_prob=False,
            with_entropy=False,
            store_state=False,
        )
        action = rst[0]
        return action

    @torch.no_grad()
    def choose_action(
        self,
        state: np.ndarray | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = as_tsr(state, device=self.device, dtype=self.dtype)  # (...,dimX)
        rst = self._integ_infer(
            state,
            greedy=False,
            with_log_prob=True,
            with_entropy=False,
            store_state=True,
        )
        action = rst[0]
        logpa = cast(torch.Tensor, rst[1])
        return action, logpa

    def _integ_infer(
        self,
        state: torch.Tensor,
        action: torch.Tensor | None = None,
        greedy: bool = False,
        with_log_prob: bool = False,
        with_entropy: bool = False,
        store_state: bool = True,
        joint: bool = False,
    ):
        """
        正向计算

        Args:
            state (torch.Tensor): 环境状态 (...,dimX)
            action (torch.Tensor | None, optional): 参考环境动作 (...,dimA). Defaults to None.
            greedy (bool, optional): (action=None时有效)需要重新选择动作时的策略 1->依分布随机, 0->依分布贪心. Defaults to False.
            with_log_prob (bool, optional): _description_. Defaults to True.
            with_entropy (bool, optional): _description_. Defaults to True.
            joint (bool, optional): 是否视为联合分布,若是则对数似然和熵都求和. Defaults to False.

        Returns:
            action (torch.Tensor): 环境动作 (...,dimA)
            log_prob (torch.Tensor|None): 动作对数似然 (...,1)
            entropy (torch.Tensor|None): 动作分布联合熵 (...,1)
        """
        if store_state:
            self.state = state  # 存储当前环境状态

        # 状态归一化
        x = (state - self.observation_min) / self._buf_obs_span  # (...,dimX)
        _shphd = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])  # (N, dimX)

        pi_ = self.actor.get_dist(x)
        if action is None:  # 需要重新采样
            if greedy:
                act_k = cast(torch.Tensor, pi_.mean)
            else:
                act_k = pi_.sample()
            act_k = torch.clip(act_k, 0, 1)

            action = self._buf_act_min + act_k * self._buf_act_span
            action = action.reshape(*_shphd, -1)  # (...,dimA)
        else:
            assert action.shape[:-1] == _shphd, ("invalid action shape", action.shape)
            act_k = (action - self.action_min) / self._buf_act_span

        if with_log_prob:
            logpa = cast(torch.Tensor, pi_.log_prob(act_k))  # (B,dimA)
            if logpa.shape[-1] > 1 and joint:
                logpa = logpa.sum(-1, keepdim=True)  # (B,1)

            logpa = logpa.reshape(*_shphd, -1)  # (...,dimA|1)
        else:
            logpa = None

        if with_entropy:
            entropy = pi_.entropy()  # (B,dimA)
            if entropy.shape[-1] > 1 and joint:
                entropy = entropy.sum(-1, keepdim=True)  # (B,1)
            entropy = entropy.reshape(*_shphd, -1)  # (...,dimA|1)
        else:
            entropy = None
        return action, logpa, entropy

    @torch.no_grad()
    def post_act(
        self,
        obs_next: torch.Tensor | np.ndarray | None,
        act: torch.Tensor | np.ndarray,  # 环境动作
        act_log_prob: torch.Tensor | np.ndarray,
        rew: torch.Tensor | np.ndarray,
        terminated: torch.Tensor | np.ndarray,
        truncated: torch.Tensor | np.ndarray,
        global_step: int,
        env_indices: torch.Tensor | None = None,
    ) -> None:
        # 张量化
        obs = self.state  # (B,dimX)
        act = as_tsr(act, device=self.device)
        act_log_prob = as_tsr(act_log_prob, device=self.device)
        terminated = as_tsr(terminated, device=self.device)
        truncated = as_tsr(truncated, device=self.device)
        rew = as_tsr(rew, device=self.device)
        if obs_next is not None:
            obs_next = as_tsr(obs_next, device=self.device)
            pass
        else:
            raise NotImplementedError("不要后继状态你评估个锤子，上天啊")

        self.returns[...] += rew
        done = terminated | truncated  # (B,1)
        if done.any():
            indices = torch.where(done)[0]

            if self.writer:
                rets = cast(np.ndarray, self.returns[indices].cpu().numpy())
                self.writer.add_scalar(
                    f"return/{self.name}_mean",
                    np.mean(rets).item(),
                    global_step=global_step,
                )
                if len(indices) > 1:
                    self.writer.add_scalar(
                        f"return/{self.name}_std",
                        np.std(rets).item(),
                        global_step=global_step,
                    )
            self.returns[indices] = 0.0

        index_store = slice(None) if env_indices is None else env_indices
        current_iteration_batch = cast(
            RolloutBatchProtocol,
            Batch(
                obs=obs[index_store],  # (B,dimX)
                act=act[index_store],  # (B,dimA)
                act_log_prob=act_log_prob,  # (B,1)
                obs_next=obs_next[index_store],  # (B,dimX)
                rew=rew[index_store],  # (B,1)
                terminated=terminated[index_store],  # (B,1)
                truncated=truncated[index_store],  # (B,1)
            ),
        )
        self.replay_buffer.add(current_iteration_batch, env_indices)

    def update(self, global_step: int):
        data_batch = self.replay_buffer.sample()
        # 张量化
        obs = as_tsr(data_batch.obs, device=self.device, dtype=self.dtype)  # (L,B,dimX)
        act = as_tsr(data_batch.act, device=self.device)
        act_log_prob = as_tsr(data_batch.act_log_prob, device=self.device)
        rew = as_tsr(data_batch.rew, device=self.device)
        obs_next = as_tsr(data_batch.obs_next, device=self.device)
        done = as_tsr(data_batch.done, device=self.device, dtype=self.dtype)  # float
        ndone = 1.0 - done

        # 归一化
        obs = normalize(obs, self.observation_min, self.observation_max)
        obs_next = normalize(obs_next, self.observation_min, self.observation_max)

        adv_freq = 1  # 重算优势函数的间隔

        # Optimize policy for K epochs:
        actor = self.actor
        for _ in range(self.repeat):
            # Calculate the advantage using GAE
            with torch.no_grad():  # adv and v_target have no gradient
                v_s: torch.Tensor = self.critic(obs)
                v_s = v_s.detach()
                v_s_prime: torch.Tensor = self.critic(obs_next)
                v_s_prime = v_s_prime.detach()

            adv = torch.zeros_like(rew)
            delta = rew + self.gamma * ndone * v_s_prime - v_s
            discount = ndone * self.gamma * self.gae_lambda
            _gae = torch.zeros_like(discount[0])
            for i in range(len(adv) - 1, -1, -1):
                _gae = delta[i] + discount[i] * _gae
                adv[i] = _gae

            v_target = adv + v_s
            if self.use_adv_norm:
                adv = (adv - adv.mean()) / (adv.std() + 1e-5)
            for index in BatchSampler(
                SubsetRandomSampler(range(len(data_batch))), self.mini_batch_size, False
            ):
                _obs = obs[index]
                _act = act[index]
                _obs2 = obs_next[index]
                _nterm2 = ndone[index]

                # Calculate the advantage using GAE
                with torch.no_grad():  # adv and v_target have no gradient
                    v_s: torch.Tensor = _nterm2 * self.critic(_obs)
                    v_s = v_s.detach()
                    v_s_prime: torch.Tensor = self.critic(obs_next)
                    v_s_prime = v_s_prime.detach()

                adv = torch.zeros_like(rew)
                delta = rew + self.gamma * ndone * v_s_prime - v_s
                discount = ndone * self.gamma * self.gae_lambda
                _gae = torch.zeros_like(discount[0])
                for i in range(len(adv) - 1, -1, -1):
                    _gae = delta[i] + discount[i] * _gae
                    adv[i] = _gae

                rst = self._integ_infer(
                    _obs,
                    _act,
                    with_log_prob=True,
                    with_entropy=True,
                    store_state=False,
                )
                logpa_new = cast(torch.Tensor, rst[1])
                dist_entropy = cast(torch.Tensor, rst[2])

                logpa_new = logpa_new.sum(-1, keepdim=True)
                dist_entropy = dist_entropy.sum(-1, keepdim=True)
                ratio = torch.exp(
                    logpa_new.sum(-1, keepdim=True)
                    - act_log_prob[index].sum(-1, keepdim=True)
                )

                surr1 = ratio * adv[index]
                surr2 = (
                    torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                )
                actor_loss = (
                    -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                )  # calculate policy entropy
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(_obs)
                critic_loss = F.mse_loss(v_target[index], v_s)

                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        # self.replay_buffer.reset()

        if self.writer:
            self.writer.add_scalar(
                f"loss/{self.name}_actor", actor_loss.mean().item(), global_step
            )
            self.writer.add_scalar(
                f"loss/{self.name}_critic", critic_loss.item(), global_step
            )

        return {
            "actor_loss": actor_loss.mean().item(),
            "critic_loss": critic_loss.item(),
        }
