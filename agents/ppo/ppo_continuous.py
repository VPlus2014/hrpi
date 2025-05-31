import torch
import torch.nn.functional as F
import numpy as np
from gymnasium import spaces
from typing import cast
from tianshou.data import Batch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from agents.modules import GaussianAcotr, Critic
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
    ):
        super().__init__(name, observation_space, action_space, batch_size, num_envs, writer, device)
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

        self.actor = GaussianAcotr(
            state_dim = observation_space.shape[0],
            action_dim = action_space.shape[0],
            action_min = self.action_min,
            action_max = self.action_max,
            hidden_sizes = actor_hidden_sizes
        ).to(device=self.device)
        self.critic = Critic(
            state_dim = observation_space.shape[0],
            hidden_sizes = critic_hidden_sizes
        ).to(device=self.device)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=self.adam_eps)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=self.adam_eps)

        self.replay_buffer = ReplayBuffer(
            state_dim = observation_space.shape[0],
            action_dim = action_space.shape[0],
            size = self.batch_size,
            num_envs = self.num_envs,
            ignore_obs_next = True,
            device = self.device
        )
        self.returns = torch.zeros(size=(self.num_envs, 1), device=self.device)

    def evaluate(self, state: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float)
        state = state.unsqueeze(dim=0)
        state = state.to(device=self.device)
        action: torch.Tensor = self.actor(state)

        action = action.squeeze(dim=0)
        
        return action

    def choose_action(self, state: np.ndarray | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float)
        self.state = state.to(device=self.device)
        # 归一化
        state_norm = normalize(self.state, self.observation_low, self.observation_high)

        with torch.no_grad():
            dist = self.actor.get_dist(state_norm)
            act = dist.sample()  # Sample the action according to the probability distribution
            act = torch.clamp(act, min=self.action_min, max=self.action_max)  # [-max,max]
            act_log_prob: torch.Tensor = dist.log_prob(act)  # The log probability density of the action
        
        return act, act_log_prob
    
    def post_act(
        self, 
        obs_next: np.ndarray | torch.Tensor | None, 
        act: np.ndarray | torch.Tensor, 
        act_log_prob: np.ndarray | torch.Tensor, 
        rew: np.ndarray | torch.Tensor, 
        terminated: np.ndarray | torch.Tensor, 
        truncated: np.ndarray | torch.Tensor,
        global_step: int,
        env_indices: torch.Tensor | None = None
    ) -> None:
        if isinstance(obs_next, np.ndarray):
            obs_next = torch.from_numpy(obs_next).to(self.replay_buffer.device)
        if isinstance(act, np.ndarray):
            act = torch.from_numpy(act).to(self.replay_buffer.device)
        if isinstance(act_log_prob, np.ndarray):
            act_log_prob = torch.from_numpy(act_log_prob).to(self.replay_buffer.device)
        if isinstance(rew, np.ndarray):
            rew = torch.from_numpy(rew).to(self.replay_buffer.device)
        if isinstance(terminated, np.ndarray):
            terminated = torch.from_numpy(terminated).to(self.replay_buffer.device)
        if isinstance(truncated, np.ndarray):
            truncated = torch.from_numpy(truncated).to(self.replay_buffer.device)

        current_iteration_batch = cast(
            RolloutBatchProtocol,
                Batch(
                    obs=self.state[env_indices],
                    act=act[env_indices],
                    act_log_prob=act_log_prob[env_indices],
                    rew=rew[env_indices],
                    terminated=terminated[env_indices],
                    truncated=truncated[env_indices],
                    done=torch.logical_or(truncated[env_indices], terminated[env_indices]),
                )
        )
        self.replay_buffer.add(current_iteration_batch, env_indices)

        self.returns[env_indices] += rew[env_indices]
        done = torch.logical_or(truncated, terminated)
        if torch.any(done):
            indices = torch.where(done)[0].to(device=torch.device("cpu"))
            self.writer.add_scalar(f"return/{self.name}_mean", self.returns[indices].mean(dim=0).item(), global_step=global_step)
            if len(indices) > 1:
                self.writer.add_scalar(f"return/{self.name}_std", self.returns[indices].std(dim=0).item(), global_step=global_step)
            self.returns[indices] = 0.0

    def update(self, global_step: int):
        data_batch = self.replay_buffer.sample()
        if isinstance(data_batch.obs, np.ndarray):
            obs = torch.from_numpy(data_batch.obs)
        elif isinstance(data_batch.obs, torch.Tensor):
            obs = data_batch.obs
        obs = obs.to(device=self.device)
        # 归一化
        obs = normalize(obs, self.observation_low, self.observation_high)

        if isinstance(data_batch.act, np.ndarray):
            act = torch.from_numpy(data_batch.act)
        elif isinstance(data_batch.act, torch.Tensor):
            act = data_batch.act
        act = act.to(device=self.device)

        if isinstance(data_batch.act_log_prob, np.ndarray):
            act_log_prob = torch.from_numpy(data_batch.act_log_prob)
        elif isinstance(data_batch.act_log_prob, torch.Tensor):
            act_log_prob = data_batch.act_log_prob
        act_log_prob = act_log_prob.to(device=self.device)

        if isinstance(data_batch.rew, np.ndarray):
            rew = torch.from_numpy(data_batch.rew)
        elif isinstance(data_batch.rew, torch.Tensor):
            rew = data_batch.rew
        rew = rew.to(device=self.device)

        if isinstance(data_batch.obs_next, np.ndarray):
            obs_next = torch.from_numpy(data_batch.obs_next)
        elif isinstance(data_batch.obs_next, torch.Tensor):
            obs_next = data_batch.obs_next
        obs_next = obs_next.to(device=self.device)
        # 归一化
        obs_next = normalize(obs_next, self.observation_low, self.observation_high)

        if isinstance(data_batch.done, np.ndarray):
            done = torch.from_numpy(data_batch.done)
        elif isinstance(data_batch.done, torch.Tensor):
            done = data_batch.done
        done = done.to(dtype=torch.float32, device=self.device)

        # Calculate the advantage using GAE
        with torch.no_grad():  # adv and v_target have no gradient
            v_s: torch.Tensor = self.critic(obs)
            v_s = v_s.detach()
            v_s_prime: torch.Tensor = self.critic(obs_next)
            v_s_prime = v_s_prime.detach()
        
        adv = torch.zeros_like(rew)
        delta = rew + self.gamma*(1.0-done)*v_s_prime - v_s
        discount = (1-done)*self.gamma*self.gae_lambda
        _gae = torch.zeros_like(discount[0])
        for i in range(len(adv)-1, -1, -1):
            _gae = delta[i] + discount[i]*_gae
            adv[i] = _gae 

        v_target = adv + v_s
        if self.use_adv_norm:
            adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize policy for K epochs:
        for _ in range(self.repeat):
            for index in BatchSampler(SubsetRandomSampler(range(len(data_batch))), self.mini_batch_size, False):
                dist = self.actor.get_dist(obs[index])
                dist_entropy = dist.entropy().sum(-1, keepdim=True)
                a_logprob_now: torch.Tensor = dist.log_prob(act[index])
                ratio = torch.exp(a_logprob_now.sum(-1, keepdim=True) - act_log_prob[index].sum(-1, keepdim=True))

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef*dist_entropy                          # calculate policy entropy
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
            self.writer.add_scalar(f"loss/{self.name}_actor", actor_loss.mean().item(), global_step)
            self.writer.add_scalar(f"loss/{self.name}_critic", critic_loss.item(), global_step)

        return {"actor_loss": actor_loss.mean().item(), "critic_loss": critic_loss.item()}
