from __future__ import annotations
from typing import TYPE_CHECKING
import torch
import torch.nn.functional as F
import numpy as np
from gymnasium import spaces
from typing import cast
from tianshou.data import Batch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

as_tsr = torch.asarray

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

from agents.modules import GaussianActor, Critic
from agents.agent import Agent
from replay_buffer.replaybuffer import ReplayBuffer
from replay_buffer.protocol import RolloutBatchProtocol

from environments.utils.space import normalize


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
        self.action_min = torch.from_numpy(action_space.low).to(device=self.device)
        self.action_max = torch.from_numpy(action_space.high).to(device=self.device)
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
            ignore_obs_next=True,
            device=self.device,
            float_dtype=self.dtype,
        )
        self.returns = torch.zeros(size=(self.num_envs, 1), device=self.device)

    def evaluate(self, state: np.ndarray | torch.Tensor) -> torch.Tensor:
        state = torch.asarray(state, dtype=self.dtype, device=self.device)  # (dimX)
        state = state.unsqueeze(dim=0)  # (1, dimX)

        action: torch.Tensor = self.actor(state)
        action = action.squeeze(dim=0)

        return action

    def _forward(
        self,
        state: torch.Tensor,
        act: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.actor.get_dist(state)
        if act is None:
            act = dist.sample()
            act = torch.clamp(act, self.action_min, self.action_max)  # [-max,max]
        act_log_prob: torch.Tensor = dist.log_prob(
            act
        )  # The log probability density of the action
        return act, act_log_prob

    @torch.no_grad()
    def choose_action(
        self,
        state: np.ndarray | torch.Tensor,
        action: np.ndarray | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.state = as_tsr(state, dtype=self.dtype, device=self.device)  # (...,dimX)
        # 归一化
        state_norm = normalize(self.state, self.observation_low, self.observation_high)

        act, act_log_prob = self._forward(state_norm)

        return act, act_log_prob

    @torch.no_grad()
    def post_act(
        self,
        obs_next: np.ndarray | torch.Tensor | None,
        act: np.ndarray | torch.Tensor,
        act_log_prob: np.ndarray | torch.Tensor,
        rew: np.ndarray | torch.Tensor,
        terminated: np.ndarray | torch.Tensor,
        truncated: np.ndarray | torch.Tensor,
        global_step: int,
        env_indices: torch.Tensor | None = None,
    ) -> None:
        # 张量化
        # if obs_next is not None:
        #     obs_next = as_tsr(obs_next, device=self.device)
        act = as_tsr(act, device=self.device)
        act_log_prob = as_tsr(act_log_prob, device=self.device)
        rew = as_tsr(rew, device=self.device)
        terminated = as_tsr(terminated, device=self.device, dtype=torch.bool)
        truncated = as_tsr(truncated, device=self.device, dtype=torch.bool)

        current_iteration_batch = cast(
            RolloutBatchProtocol,
            Batch(
                obs=self.state[env_indices],
                act=act[env_indices],
                act_log_prob=act_log_prob[env_indices],
                rew=rew[env_indices],
                terminated=terminated[env_indices],
                truncated=truncated[env_indices],
                done=truncated[env_indices] | terminated[env_indices],
            ),
        )
        self.replay_buffer.add(current_iteration_batch, env_indices)

        self.returns[env_indices] += rew[env_indices]
        done = terminated | truncated
        if torch.any(done):
            indices = torch.where(done)[0]
            if self.writer:
                rets = self.returns[indices].cpu()
                self.writer.add_scalar(
                    f"return/{self.name}_mean",
                    rets.mean(dim=0).item(),
                    global_step=global_step,
                )
                if len(indices) > 1:
                    self.writer.add_scalar(
                        f"return/{self.name}_std",
                        rets.std(dim=0).item(),
                        global_step=global_step,
                    )
            self.returns[indices] = 0.0

    def update(self, global_step: int):
        data_batch = self.replay_buffer.sample()
        # 张量化
        obs = as_tsr(data_batch.obs, device=self.device, dtype=self.dtype)
        act = as_tsr(data_batch.act, device=self.device)
        act_log_prob = as_tsr(data_batch.act_log_prob, device=self.device)
        rew = as_tsr(data_batch.rew, device=self.device)
        obs_next = as_tsr(data_batch.obs_next, device=self.device)
        done = as_tsr(data_batch.done, device=self.device, dtype=self.dtype)  # float
        ndone = 1.0 - done

        # 归一化
        obs = normalize(obs, self.observation_low, self.observation_high)
        obs_next = normalize(obs_next, self.observation_low, self.observation_high)

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

        # Optimize policy for K epochs:
        for _ in range(self.repeat):
            for index in BatchSampler(
                SubsetRandomSampler(range(len(data_batch))), self.mini_batch_size, False
            ):
                dist = self.actor.get_dist(obs[index])
                dist_entropy = dist.entropy().sum(-1, keepdim=True)
                a_logprob_now: torch.Tensor = dist.log_prob(act[index])
                ratio = torch.exp(
                    a_logprob_now.sum(-1, keepdim=True)
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

                v_s = self.critic(obs[index])
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
