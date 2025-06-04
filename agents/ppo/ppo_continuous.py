from __future__ import annotations
import os

os.environ["NUMEXPR_MAX_THREADS"] = str(os.cpu_count())
import logging
from typing import TYPE_CHECKING, TypeVar
import torch
import torch.nn.functional as F
import numpy as np
from gymnasium import spaces
from typing import cast
from tianshou.data import Batch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.nn.utils import clip_grad_norm_, clip_grad_value_


as_tsr = torch.asarray
_DEBUG = True


def as_np(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.asarray(x)


if TYPE_CHECKING:
    from torch.utils.tensorboard.writer import SummaryWriter

from agents.modules import GaussianActor, Critic
from agents.agent import Agent
from replay_buffer.replaybuffer import ReplayBuffer
from replay_buffer.protocol import RolloutBatchProtocol
from replay_buffer.trajbuffer import RETrajReplayBuffer

from environments.utils.space import normalize, affcmb


class PPOContinuous(Agent):

    def __init__(
        self,
        name: str,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        buffer_size: int,
        learn_batch_size: int,
        mini_batch_size: int,
        lr_a: float,
        actor_hidden_sizes: list[int],
        lr_c: float,
        critic_hidden_sizes: list[int],
        gamma: float,
        epsilon: float,
        policy_entropy_coef: float,
        use_grad_clip: bool,
        use_adv_norm: bool,
        repeat: int,
        adam_eps: float = 1e-8,
        num_envs: int = 1,
        env_out_torch=True,
        gae_lambda: float = 0.8,
        gae_horizon: int = 100,
        gae_forward=False,
        writer: SummaryWriter | None = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float,
        max_steps: int = 1000,
        np_float=np.float32,
        np_action_dtype=np.float32,
        logr: logging.Logger = Agent.logr,
    ):
        self.logr = logr
        super().__init__(
            name,
            observation_space,
            action_space=action_space,
            buffer_size=buffer_size,
            batch_size=learn_batch_size,
            num_envs=num_envs,
            writer=writer,
            device=device,
            dtype=dtype,
        )
        if mini_batch_size > learn_batch_size:
            mini_batch_size = learn_batch_size
            logr.info(
                (
                    "mini_batch_size > batch_size, mini_batch_size set to",
                    learn_batch_size,
                )
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
            logr=logr.getChild("actor"),
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

        use_re_buffer = True
        if use_re_buffer:
            self.replay_buffer = RETrajReplayBuffer(
                obs_shape=observation_space.shape,
                act_shape=action_space.shape,
                max_steps=max_steps,
                max_trajs=buffer_size,
                num_envs=num_envs,
                float_dtype=np_float,
                action_dtype=np_action_dtype,
                logr=logr.getChild("replay_buffer"),
            )
        else:
            raise NotImplementedError
            self.replay_buffer = ReplayBuffer(
                state_dim=observation_space.shape[0],
                action_dim=action_space.shape[0],
                size=buffer_size,
                num_envs=num_envs,
                compact=True,
                device=self.device,
                float_dtype=self.dtype,
            )

        self.returns = np.zeros((self.num_envs, 1))

        import inspect

        frame = inspect.currentframe()
        if frame is not None:
            arg_names, _, _, values = inspect.getargvalues(frame)
            params = {name: values[name] for name in arg_names}
            del frame  # 避免循环引用导致内存泄漏
            logr.debug(("init with", params))

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
        logr = self.logr
        if store_state:
            self.state = state  # 存储当前环境状态

        # 状态归一化
        x = (state - self.observation_min) / self._buf_obs_span  # (...,dimX)
        _shphd = x.shape[:-1]
        # x = x.reshape(-1, x.shape[-1])  # (N, dimX)
        if _DEBUG:
            xinf = x.isinf()
            xnan = x.isnan()
            anyinf = xinf.any().item()
            anynan = xnan.any().item()

            logr.debug(
                {
                    "X": x.mean().item(),
                    "Xstd": x.std().item(),
                    "X not finite": anyinf or anynan,
                    "X any inf": anyinf,
                    "X any nan": anynan,
                }
            )
            assert not anyinf, ("x inf", torch.where(xinf)[0])
            assert not anynan, ("x nan", torch.where(xnan)[0])

        pi_ = self.actor.get_dist(x)  # @forward
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
            assert (act_k >= 0).all(), "expect a_k>=0"
            assert (act_k <= 1).all(), "expect a_k<=1"

        if with_log_prob:
            logpa = cast(torch.Tensor, pi_.log_prob(act_k))  # (...,dimA)
            if logpa.shape[-1] > 1 and joint:
                logpa = logpa.sum(-1, keepdim=True)  # (...,1)

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
        obs: np.ndarray | torch.Tensor,
        act: torch.Tensor | np.ndarray,  # 环境动作
        act_log_prob: torch.Tensor | np.ndarray,
        obs_next: torch.Tensor | np.ndarray,
        rew: torch.Tensor | np.ndarray,
        terminated: torch.Tensor | np.ndarray,
        truncated: torch.Tensor | np.ndarray,
        global_step: int,
        final_obs: torch.Tensor | np.ndarray,
        env_indices: torch.Tensor | None = None,
    ) -> None:
        """收集数据"""
        use_torch = False
        if use_torch:
            raise NotImplementedError
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
                raise NotImplementedError("!!!")
        else:
            pass
        obs = as_np(obs)
        act = as_np(act)
        rew = as_np(rew)
        terminated = as_np(terminated)
        truncated = as_np(truncated)
        act_log_prob = as_np(act_log_prob)
        final_obs = as_np(final_obs)
        done = terminated | truncated

        obs2 = np.where(done, final_obs, obs_next)  # 真正的后继状态

        index_store = slice(None) if env_indices is None else env_indices
        if not use_torch and isinstance(index_store, torch.Tensor):
            index_store = as_np(index_store)
        current_iteration_batch = cast(
            RolloutBatchProtocol,
            Batch(
                obs=obs[index_store],  # (B,dimX)
                act=act[index_store],  # (B,dimA)
                act_log_prob=act_log_prob[index_store],  # (B,1)
                obs_next=obs2[index_store],  # (B,dimX)
                rew=rew[index_store],  # (B,1)
                terminated=terminated[index_store],  # (B,1)
                truncated=truncated[index_store],  # (B,1)
            ),
        )
        if isinstance(self.replay_buffer, RETrajReplayBuffer):
            self.replay_buffer.add(
                obs=cast(np.ndarray, current_iteration_batch.obs),
                act=cast(np.ndarray, current_iteration_batch.act),
                obs_next=cast(np.ndarray, current_iteration_batch.obs_next),
                rew=cast(np.ndarray, current_iteration_batch.rew),
                term=cast(np.ndarray, current_iteration_batch.terminated),
                trunc=cast(np.ndarray, current_iteration_batch.truncated),
                act_log_prob=cast(np.ndarray, current_iteration_batch.act_log_prob),
            )
        else:
            raise NotImplementedError

        # 统计
        self.returns[...] += as_np(rew)
        done = as_np(terminated | truncated).ravel()  # (B,)
        if done.any():
            indices = np.where(done)[0]

            if self.writer:
                rets = cast(np.ndarray, self.returns[indices])
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

    def update(self, global_step: int):
        logr = self.logr
        logr.debug("start update")
        data_batch = self.replay_buffer.sample(batch_size=self.batch_size)
        # 张量化
        obs = as_tsr(
            data_batch.obs, device=self.device, dtype=self.dtype
        )  # (L+1,B,dimX)
        act = as_tsr(data_batch.act, device=self.device)  # (L,B,dimA)
        L = act.shape[0]
        B = act.shape[1]
        assert obs.shape[0] == L + 1 and obs.shape[1] == B, (
            "invalid obs shape",
            obs.shape,
        )
        act_log_prob = as_tsr(data_batch.act_log_prob, device=self.device).reshape(
            L, B, -1
        )
        if act_log_prob.shape[-1] > 1:
            act_log_prob = act_log_prob.sum(-1, keepdim=True)  # (L,B,1)
        rew = as_tsr(data_batch.rew, device=self.device).reshape(L, B, 1)  # (L,B,1)
        term = as_tsr(
            data_batch.terminated, device=self.device, dtype=torch.bool
        ).reshape(L + 1, B, 1)
        noterm = ~term  # (L+1,B,1)
        noterm1 = noterm[:-1]  # (L,B,1)
        term_f = term.to(dtype=self.dtype)  # float
        # trunc = as_tsr(data_batch.truncated, device=self.device, dtype=torch.bool)
        # trunc_f = trunc.to(dtype=self.dtype)  # float
        noterm_f = 1.0 - term_f

        if _DEBUG:
            term_inc = term[:-1] <= term[1:]  # 单调性
            term_inc_all = term_inc.all().item()
            logr.debug(
                {
                    "rew_mean": rew.mean().item(),
                    "rew_std": rew.std().item(),
                    "rew_min": rew.min().item(),
                    "rew_max": rew.max().item(),
                    "term_inc": term_inc_all,
                }
            )
            assert term_inc_all, "term not temporally monotonic"

        # 归一化
        obs = normalize(obs, self.observation_min, self.observation_max)

        adv_freq = 2  # 重算优势函数的间隔

        # Optimize policy for K epochs:
        actor = self.actor  # @update
        for _ in range(self.repeat):
            # Calculate the advantage using GAE
            # adv and v_target have no gradient
            if (_ == 0) or _ % adv_freq == 0:
                with torch.no_grad():
                    hVs = cast(torch.Tensor, noterm_f * self.critic(obs))  # (h\odot V)
                    X1 = obs[:-1]
                    hV2p = hVs[1:]
                    hV1p = hVs[:-1]
                    hV1t = rew + self.gamma * hV2p

                    gl = self.gamma * self.gae_lambda
                    delta = hV1t - hV1p  # (L,B,1)
                    adv = delta.clone()  # (L,B,1)
                    for t in reversed(range(L - 1)):  # 后向GAE
                        adv[t] = delta[t] + gl * adv[t + 1]

                    v_target = hV1p + adv  # lambda 估计

                    if self.use_adv_norm:
                        adv = (adv - adv.mean()) / (adv.std() + 1e-5)

                    if _DEBUG:
                        logr.debug(
                            {
                                "vs": hVs[noterm].mean().item(),
                                "lambda_v_target": v_target[noterm1].mean().item(),
                                "adv": adv[noterm1].mean().item(),
                            }
                        )

            for index in BatchSampler(
                SubsetRandomSampler(range(len(data_batch))), self.mini_batch_size, False
            ):
                _obs = X1[index]
                _act = act[index]
                _adv = adv[index]

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
                ratio = torch.exp(logpa_new.sum(-1, keepdim=True) - act_log_prob[index])

                if _DEBUG:
                    logr.debug(
                        {
                            "ratio-1": ratio.mean().item() - 1,
                            "logpa_new": logpa_new.mean().item(),
                            "dist_entropy": dist_entropy.mean().item(),
                        }
                    )

                surr1 = ratio * _adv
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * _adv
                actor_loss = (
                    -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                )  # calculate policy entropy
                actor_loss = actor_loss.mean()
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                if self.use_grad_clip:
                    clip_grad_value_(actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(_obs)
                critic_loss = F.mse_loss(v_s, v_target[index])

                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:
                    clip_grad_value_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        # self.replay_buffer.reset()

        actor_loss_ = actor_loss.item()
        critic_loss_ = critic_loss.item()

        sw = self.writer
        if sw:
            name = self.name
            dr = cast(torch.Tensor, ratio - 1)
            sw.add_scalar(f"actor/loss/{name}", actor_loss_, global_step)
            sw.add_scalar(f"critic/loss/{name}", critic_loss_, global_step)
            sw.add_scalar(
                f"actor/entropy/{name}/mean", dist_entropy.mean().item(), global_step
            )
            sw.add_scalar(
                f"actor/entropy/{name}/std", dist_entropy.std().item(), global_step
            )
            sw.add_scalar(f"actor/ratio-1/{name}/mean", dr.mean().item(), global_step)
            sw.add_scalar(f"actor/ratio-1/{name}/std", dr.std().item(), global_step)
            sw.add_scalar(f"critic/adv/{name}/GAE_mean", adv.mean().item(), global_step)
            sw.add_scalar(f"critic/adv/{name}/GAE_std", adv.std().item(), global_step)
            sw.add_scalar(
                f"critic/adv/{name}/vanilla_mean", delta.mean().item(), global_step
            )
            sw.add_scalar(
                f"critic/adv/{name}/vanilla_std", delta.std().item(), global_step
            )
            sw.add_scalar(
                f"critic/vs/{name}/lambda_mean", v_target.mean().item(), global_step
            )
            sw.add_scalar(
                f"critic/vs/{name}/lambda_std", v_target.std().item(), global_step
            )
            sw.add_scalar(
                f"critic/vs/{name}/vanilla_mean", hV1t.mean().item(), global_step
            )
            sw.add_scalar(
                f"critic/vs/{name}/vanilla_std", hV1t.std().item(), global_step
            )
            sw.add_scalar(
                f"critic/vs/{name}/pred_mean", hV1p.mean().item(), global_step
            )
            sw.add_scalar(f"critic/vs/{name}/pred_std", hV1p.std().item(), global_step)
            sw.add_scalar(
                f"critic/lr/{name}",
                self.optimizer_critic.param_groups[0]["lr"],
                global_step,
            )
            sw.add_scalar(
                f"critic/lr/{name}/actor",
                self.optimizer_actor.param_groups[0]["lr"],
                global_step,
            )
        if _DEBUG:
            logr.debug(
                {
                    "actor_loss": actor_loss_,
                    "critic_loss": critic_loss_,
                }
            )

        return {
            "actor_loss": actor_loss_,
            "critic_loss": critic_loss_,
        }
