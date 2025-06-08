from __future__ import annotations
from typing import TYPE_CHECKING
import torch
import torch.nn.functional as F
import numpy as np
from typing import cast
from tianshou.data import Batch
from collections import OrderedDict
from gymnasium import spaces
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter
from functools import cached_property
from environments.utils.space import get_spaces_shape, space2box, flatten, unflatten
from environments.utils.math_pt import (
    Qx,
    Qy,
    Qz,
    quat_rotate,
    quat_rotate_inv,
    quat_mul,
    lerp,
    nlerp,
    herp,
    aer2ned,
)
from ..modules import GaussianActor, Critic
from ..agent import Agent
from ..ppo.ppo_continuous import PPOContinuous
from replay_buffer.replaybuffer import ReplayBuffer
from replay_buffer.protocol import RolloutBatchProtocol
from copy import deepcopy

as_tsr = torch.asarray


class HPPOContinuous(Agent):
    def __init__(
        self,
        env_observation_space_dict: spaces.Dict,
        env_action_space_dict: spaces.Dict,
        h_batch_size: int,
        h_mini_batch_size: int,
        h_lr_a: float,
        h_actor_hidden_sizes: list[int],
        h_lr_c: float,
        h_critic_hidden_sizes: list[int],
        h_gamma: float,
        h_gae_lambda: float,
        h_epsilon: float,
        h_policy_entropy_coef: float,
        h_use_grad_clip: bool,
        h_use_adv_norm: bool,
        h_repeat: int,
        h_adam_eps: float,
        l_batch_size: int,
        l_mini_batch_size: int,
        l_lr_a: float,
        l_actor_hidden_sizes: list[int],
        l_lr_c: float,
        l_critic_hidden_sizes: list[int],
        l_gamma: float,
        l_gae_lambda: float,
        l_epsilon: float,
        l_policy_entropy_coef: float,
        l_use_grad_clip: bool,
        l_use_adv_norm: bool,
        l_repeat: int,
        l_adam_eps: float,
        temporal_extension: int = 1,
        num_envs: int = 1,
        writer: SummaryWriter | None = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float,  # torch浮点类型
        np_float=np.float32,  # numpy浮点类型
    ):
        self.temporal_extension = temporal_extension
        self.env_observation_space_dict = env_observation_space_dict
        self.env_action_space_dict = env_action_space_dict

        # 定义高层状态空间及动作空间
        self.high_observation_space_dict = deepcopy(env_observation_space_dict)
        self.high_observation_space_dict["pre_expected_aircraft_position_g"] = (
            self.env_observation_space_dict["aircraft_position_g"]
        )

        self.high_action_space_dict = spaces.Dict()
        self.high_action_space_dict["delta_aircraft_position_g"] = spaces.Box(
            low=-500, high=500, shape=(3,), dtype=np_float
        )

        # 定义低层状态空间及动作空间
        self.low_observation_space_dict = deepcopy(env_observation_space_dict)
        self.low_observation_space_dict["delta_aircraft_position_g"] = (
            self.high_action_space_dict["delta_aircraft_position_g"]
        )

        self.low_action_space_dict = deepcopy(env_action_space_dict)

        super().__init__(
            name="hppo",
            observation_space=space2box(
                self.high_observation_space_dict, dtype=np_float
            ),
            action_space=space2box(self.low_action_space_dict, dtype=np_float),
            buffer_size=h_batch_size,
            num_envs=num_envs,
            writer=writer,
            device=device,
            dtype=dtype,
        )

        # 创建高层智能体
        self.planner = PPOContinuous(
            name="planner",
            observation_space=space2box(
                self.high_observation_space_dict, dtype=np_float
            ),
            action_space=space2box(self.high_action_space_dict, dtype=np_float),
            learn_batch_size=h_batch_size,
            mini_batch_size=h_mini_batch_size,
            lr_a=h_lr_a,
            actor_hidden_sizes=h_actor_hidden_sizes,
            lr_c=h_lr_c,
            critic_hidden_sizes=h_critic_hidden_sizes,
            gamma=h_gamma,
            gae_lambda=h_gae_lambda,
            epsilon=h_epsilon,
            policy_entropy_coef=h_policy_entropy_coef,
            use_grad_clip=h_use_grad_clip,
            use_adv_norm=h_use_adv_norm,
            repeat=h_repeat,
            adam_eps=h_adam_eps,
            num_envs=num_envs,
            writer=writer,
            device=device,
            dtype=dtype,
        )

        # 创建低层智能体
        self.tracker = PPOContinuous(
            name="tracker",
            observation_space=space2box(self.low_observation_space_dict),
            action_space=space2box(self.low_action_space_dict),
            learn_batch_size=l_batch_size,
            mini_batch_size=l_mini_batch_size,
            lr_a=l_lr_a,
            actor_hidden_sizes=l_actor_hidden_sizes,
            lr_c=l_lr_c,
            critic_hidden_sizes=l_critic_hidden_sizes,
            gamma=l_gamma,
            gae_lambda=l_gae_lambda,
            epsilon=l_epsilon,
            policy_entropy_coef=l_policy_entropy_coef,
            use_grad_clip=l_use_grad_clip,
            use_adv_norm=l_use_adv_norm,
            repeat=l_repeat,
            adam_eps=l_adam_eps,
            num_envs=num_envs,
            writer=writer,
            device=device,
        )

        self._step = torch.zeros(
            (self.num_envs, 1), dtype=torch.int64, device=self.device
        )
        self.__rew = torch.zeros((self.num_envs, 1), device=self.device)

    def _forward(
        self, env_obs: torch.Tensor, act: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def choose_action(
        self,
        env_obs: np.ndarray | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        env_obs = as_tsr(env_obs, dtype=self.dtype, device=self.device)
        env_obs_dict, _ = unflatten(self.env_observation_space_dict, env_obs)

        # 1.计算高层智能体所需的高层观测状态
        try:
            pre_h_act_dict, _ = unflatten(self.high_action_space_dict, self.pre_h_act)
        except AttributeError:  # 没有 .pre_h_act
            pre_expected_aircraft_position_g = env_obs_dict[
                "aircraft_position_g"
            ]  # 当前位置 ENU
            self.expected_aircraft_position_g = env_obs_dict[
                "aircraft_position_g"
            ].clone()
        else:
            pre_expected_aircraft_position_g = self.expected_aircraft_position_g
        finally:
            h_obs_dict = deepcopy(env_obs_dict)
            h_obs_dict["pre_expected_aircraft_position_g"] = (
                pre_expected_aircraft_position_g
            )
            h_obs = flatten(self.high_observation_space_dict, h_obs_dict)

        # 2.高层智能体决策
        h_act, h_act_log_prob = self.planner.choose_action(h_obs)

        # 3.插值
        try:
            self.h_indices = torch.where(self._step % self.temporal_extension == 0)[0]
            self.pre_h_act[self.h_indices] = self.h_act[self.h_indices]
        except AttributeError:
            pre_h_act_dict = OrderedDict()
            pre_h_act_dict["delta_aircraft_position_g"] = torch.zeros(
                (self.num_envs, 3), device=self.device
            )
            self.pre_h_act = flatten(self.high_action_space_dict, pre_h_act_dict)
            # 记录高层策略的状态、动作
            self.pre_h_obs = h_obs
            self.h_obs = h_obs
            self.h_act = h_act
            self.h_act_log_prob = h_act_log_prob
        else:
            pre_h_act_dict, _ = unflatten(self.high_action_space_dict, self.pre_h_act)
            # 记录高层策略的状态、动作
            self.pre_h_obs[self.h_indices] = self.h_obs[self.h_indices]
            self.h_obs[self.h_indices] = h_obs[self.h_indices]
            self.h_act[self.h_indices] = h_act[self.h_indices]
            self.h_act_log_prob[self.h_indices] = h_act_log_prob[self.h_indices]
        finally:
            h_act_dict, _ = unflatten(self.high_action_space_dict, self.h_act)
            self.expected_aircraft_position_g[self.h_indices] = (
                pre_expected_aircraft_position_g[self.h_indices]
                + h_act_dict["delta_aircraft_position_g"][self.h_indices]
            )

        # 4.计算低层智能体所需的低层观测状态
        l_obs_dict = deepcopy(env_obs_dict)
        l_obs_dict["delta_aircraft_position_g"] = (
            self.expected_aircraft_position_g - env_obs_dict["aircraft_position_g"]
        )
        l_obs = flatten(self.low_observation_space_dict, l_obs_dict)

        # 5.低层智能体决策
        l_act, l_act_log_prob = self.tracker.choose_action(l_obs)

        self._step += 1
        return l_act, l_act_log_prob

    def post_act(
        self,
        env_obs: np.ndarray | torch.Tensor,
        env_obs_next: np.ndarray | torch.Tensor,
        act: np.ndarray | torch.Tensor,
        act_log_prob: np.ndarray | torch.Tensor,
        env_rew: np.ndarray | torch.Tensor,
        terminated: np.ndarray | torch.Tensor,
        truncated: np.ndarray | torch.Tensor,
        global_step: int,
        env_indices: torch.Tensor | None = None,
    ) -> None:
        env_obs = as_tsr(env_obs, device=self.device)
        env_obs_next = as_tsr(env_obs_next, device=self.device)
        act = as_tsr(act, device=self.device)
        act_log_prob = as_tsr(act_log_prob, device=self.device)
        env_rew = as_tsr(env_rew, device=self.device)
        terminated = as_tsr(terminated, dtype=torch.bool, device=self.device)
        truncated = as_tsr(truncated, dtype=torch.bool, device=self.device)

        env_obs_dict, _ = unflatten(self.env_observation_space_dict, env_obs)
        env_obs_next_dict, _ = unflatten(self.env_observation_space_dict, env_obs_next)

        # 计算位置差
        # pre_aircraft_position_difference = self.expected_aircraft_position_g - env_obs_dict["aircraft_position_g"]
        aircraft_position_g = env_obs_next_dict["aircraft_position_g"]
        aircraft_position_difference = (
            self.expected_aircraft_position_g - aircraft_position_g
        )
        # rew = 1e-1*(torch.norm(pre_aircraft_position_difference, p=2, dim=-1, keepdim=True)-torch.norm(cur_aircraft_position_difference, p=2, dim=-1, keepdim=True))
        rew = -1e-4 * torch.norm(
            aircraft_position_difference, p=2, dim=-1, keepdim=True
        )

        done = terminated | truncated
        if torch.any(done):
            indices = torch.where(done)[0]
            rew[indices] = 0.0
            self.h_indices = torch.unique(torch.cat([self.h_indices, indices]))

        # 存储低层智能体数据
        self.tracker.post_act(
            obs_next=env_obs_next,
            act=act,
            act_log_prob=act_log_prob,
            rew=rew,
            terminated=terminated,
            truncated=truncated,
            global_step=global_step,
        )

        # 存储高层智能体数据
        self.__rew += env_rew + rew
        self.planner.post_act(
            obs_next=None,
            act=self.h_act,
            act_log_prob=self.h_act_log_prob,
            rew=self.__rew,
            terminated=terminated,
            truncated=truncated,
            global_step=global_step,
            index_store=self.h_indices,
        )
        self.__rew[self.h_indices] = 0.0

        if torch.any(done):
            self._step[indices] = 0

            h_obs_dict = deepcopy(env_obs_dict)
            h_obs_dict["pre_expected_aircraft_position_g"] = env_obs_dict[
                "aircraft_position_g"
            ]
            h_obs = flatten(self.high_observation_space_dict, h_obs_dict)
            self.pre_h_obs[indices] = h_obs[indices]
            self.h_obs[indices] = h_obs[indices]
            self.expected_aircraft_position_g[indices] = env_obs_next_dict[
                "aircraft_position_g"
            ][indices]

            pre_h_act_dict = OrderedDict()
            pre_h_act_dict["delta_aircraft_position_g"] = torch.zeros(
                (self.num_envs, 3), device=self.device
            )
            pre_h_act = flatten(self.high_action_space_dict, pre_h_act_dict)
            self.pre_h_act[indices] = pre_h_act[indices]
            self.h_act[indices] = pre_h_act[indices]

    def update(self, global_step: int):
        self.tracker.update(global_step)
        self.planner.update(global_step)
