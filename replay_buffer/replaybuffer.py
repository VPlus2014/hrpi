import torch
import numpy as np
from typing import cast
from tianshou.data import Batch
from copy import deepcopy

from .protocol import RolloutBatchProtocol


class ReplayBuffer:
    """张量型经验回放池"""

    _INPUT_KEYS = (
        "obs",
        "obs_next",
        "rew",
        "truncated",
        "terminated",
        "act",
        "act_log_prob",
        "done",
    )

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        size: int,
        num_envs: int,
        ignore_obs_next: bool = False,  # 1:不存储后继状态(内存高效型)
        device: torch.device = torch.device("cpu"),
        float_dtype: torch.dtype = torch.float32,
        int_dtype: torch.dtype = torch.int64,
        state_dtype: torch.dtype | None = None,  # 默认为 float_dtype
        action_dtype: torch.dtype | None = None,  # 默认为 float_dtype
    ):

        self.max_size = int(size)
        self.num_envs = num_envs
        self.ignore_obs_next = ignore_obs_next
        self.device = device

        self._indices = torch.arange(self.max_size, device=device).repeat(
            self.num_envs, 1
        )
        if state_dtype is None:
            state_dtype = float_dtype
        if action_dtype is None:
            action_dtype = float_dtype

        obs = torch.zeros(
            (self.max_size, self.num_envs, state_dim),
            dtype=state_dtype,
            device=device,
        )
        if not self.ignore_obs_next:
            obs_next = torch.zeros(
                (self.max_size, self.num_envs, state_dim),
                dtype=state_dtype,
                device=device,
            )
        act = torch.zeros(
            (self.max_size, self.num_envs, action_dim),
            dtype=float_dtype,
            device=self.device,
        )
        rew = torch.zeros(
            (self.max_size, self.num_envs, 1), dtype=float_dtype, device=device
        )
        truncated = torch.zeros(
            (self.max_size, self.num_envs, 1), dtype=torch.bool, device=device
        )
        terminated = torch.zeros(
            (self.max_size, self.num_envs, 1), dtype=torch.bool, device=device
        )

        act_log_prob = torch.zeros(
            (self.max_size, self.num_envs, action_dim),
            dtype=float_dtype,
            device=device,
        )

        done = torch.zeros(
            (self.max_size, self.num_envs, 1), dtype=torch.bool, device=device
        )

        if not self.ignore_obs_next:
            self._meta = cast(
                RolloutBatchProtocol,
                Batch(
                    obs=obs,
                    obs_next=obs_next,
                    rew=rew,
                    truncated=truncated,
                    terminated=terminated,
                    act=act,
                    act_log_prob=act_log_prob,
                    done=done,
                ),
            )
        else:
            self._meta = cast(
                RolloutBatchProtocol,
                Batch(
                    obs=obs,
                    rew=rew,
                    truncated=truncated,
                    terminated=terminated,
                    act=act,
                    act_log_prob=act_log_prob,
                    done=done,
                ),
            )

        self.reset()

    def reset(self) -> None:
        self.last_index = torch.zeros(
            size=(self.num_envs,), dtype=torch.int64, device=self.device
        )
        self._index = torch.zeros(
            size=(self.num_envs,), dtype=torch.int64, device=self.device
        )
        self.size = torch.zeros(
            size=(self.num_envs,), dtype=torch.int64, device=self.device
        )  # size 目前没有用！！！

    def _add_index(self, env_indices: torch.Tensor | None = None):
        if env_indices is None:
            self.last_index = ptr = self._index.clone()
            self.size = torch.clamp(self.size + 1, max=self.max_size)
            self._index = torch.fmod(self._index + 1, self.max_size)
            return ptr
        else:
            self.last_index = ptr = self._index.clone()
            self.size[env_indices] = torch.clamp(
                self.size[env_indices] + 1, max=self.max_size
            )
            self._index[env_indices] = torch.fmod(
                self._index[env_indices] + 1, self.max_size
            )
            return ptr[env_indices]

    def add(
        self, batch: RolloutBatchProtocol, env_indices: torch.Tensor | None = None
    ) -> None:
        # if env_indices is not None and len(env_indices) > 0:
        _batch = cast(RolloutBatchProtocol, Batch())
        _input_keys = set(self._INPUT_KEYS)
        if self.ignore_obs_next:
            _input_keys.remove("obs_next")
        for key in _input_keys.intersection(batch.get_keys()):
            _batch[key] = batch[key]
        assert _input_keys.issubset(_batch.get_keys()), print(_batch.get_keys())

        ptr = self._add_index(env_indices)
        if env_indices is None:
            self._meta[ptr] = _batch
        elif len(env_indices) > 0:
            try:
                self._meta[ptr, env_indices] = _batch
            except RuntimeError:
                pass

    def sample(self) -> RolloutBatchProtocol:
        data_batch = deepcopy(self._meta)
        data_batch.done[self.last_index] = True
        if self.ignore_obs_next:
            data_batch.obs_next = torch.cat(
                [data_batch.obs[1:, ...], data_batch.obs[:1, ...]], dim=0
            )
        return data_batch
