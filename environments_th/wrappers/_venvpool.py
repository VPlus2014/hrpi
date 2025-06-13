from __future__ import annotations
from dataclasses import dataclass
from multiprocessing import RLock, Queue
from typing import Any, Callable, TYPE_CHECKING, Generic

import gymnasium
import numpy as np
import torch

if TYPE_CHECKING:
    from ..proto4venv import TorchSyncVecEnv
    from gymnasium.core import ObsType, ActType


class EnvInPool(gymnasium.Env[ObsType, ActType]):
    def __init__(self, env_id: int, parent: AsyncEnvPool, use_network=False) -> None:
        self._parent = parent
        self._env_idx = env_id
        raise NotImplementedError("TODO: init env")

    def step(self, action: ActType):
        tsk = self._parent.send(self._env_idx, np.asarray(action).reshape(1, -1))
        rst = tsk.wait()
        return rst

    def reset(self, **kwargs):
        tsk = self._parent.send(self._env_idx, None, **kwargs)
        rst = tsk.wait()
        return rst

    def render(self, **kwargs) -> None:
        self._parent.render(self._env_idx, **kwargs)

    def close(self) -> None:
        pass


class AsyncEnvPool:

    def __init__(self, venv: TorchSyncVecEnv, use_network=False) -> None:
        """
        将原向量化环境重新打散为多个单环境, 为兼容标准RL算法框架做适配
        TODO: 按照现行逻辑, Pool必须实现异步向量化, 这样就必须要通过 网络/跨进程通信 来实现

        Args:
            venv (CUDASyncVEnv): _description_
            use_network (bool, optional): _description_. Defaults to False.
        """
        assert not use_network, NotImplementedError(
            "CUDAEnvPool does not support use_network"
        )
        _imp = False
        assert _imp, "TODO: implement CUDAEnvPool"
        self._core = venv
        cap = self.capacity
        self._used = np.zeros(cap, dtype=np.bool_)
        self._lock = RLock()
        self._qa = [Queue(maxsize=1) for _ in range(cap)]  # proxy->core
        self._qy = [Queue(maxsize=1) for _ in range(cap)]  # core->proxy
        self._proxy = [
            EnvInPool[np.ndarray, np.ndarray](env_id=i, parent=self)
            for i in range(self.capacity)
        ]

    @property
    def capacity(self) -> int:
        return self._core.num_envs

    def new_env(self) -> EnvInPool[np.ndarray, np.ndarray]:
        with self._lock:
            idx = self._used.argmin()
            assert not self._used[idx], "All envs are in use"
            self._used[idx] = True
        return self._grp[idx].env

    def send(self, env_idx: int, action: np.ndarray | None):
        core = self._core
        buf = self._grp
        buf[env_idx].acts.append(action.reshape(1, -1))
        return rst

    def reset(self, env_idx: int, **kwargs) -> np.ndarray:
        with self._lock:
            rst = self._core.reset(env_idx, **kwargs)
        return rst

    def render(self, env_idx: int, mode: str = "human") -> None:
        self._core.render(env_idx, mode)
