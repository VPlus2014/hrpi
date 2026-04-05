from __future__ import annotations
from gymnasium.vector import VectorEnv, SyncVectorEnv
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, Union
from abc import ABC, abstractmethod
from functools import cached_property

import gymnasium
from gymnasium.utils.seeding import np_random

from numpy import ndarray
import numpy as np
from numpy.typing import NDArray
import logging
from torch.utils.tensorboard.writer import SummaryWriter

if TYPE_CHECKING:
    from typing import Any, List, Optional, Sequence
    from .simulators.proto4model import BaseModelGroup
    from ..utils.math_np import Float_NDArr, Int_NDArr, BoolNDArr
    from ..utils.log_ext import LogConfig
    from .reward_fns.proto4rf import BaseRewardFn

EnvIndexType = Union[ndarray, NDArray[np.intp], list[int], tuple[int], slice]
EnvMaskType = Union[
    ndarray, NDArray[np.bool_], list[bool], tuple[bool], type(Ellipsis), slice
]

_SliceAll = slice(None)

NestedNPDict = Dict[str, Union[NDArray, "NestedNPDict"]]


class NPSyncVecEnv(SyncVectorEnv, ABC):

    DEBUG: bool = False
    logger: logging.Logger = logging.getLogger(__name__)

    OPTION_KEY_MASK = "env_mask"
    INFO_KEY_FINAL_OBS = "final_observation"
    INFO_KEY_FINAL_INFO = "final_info"

    observations: NDArray[np.generic] | None
    """buffered observations from reset/step, shape=(num_envs, *obs_shape)"""
    _actions: NDArray[np.generic] | None
    """buffered actions, shape=(num_envs, *act_shape)"""

    def __init__(
        self,
        num_envs: int,
        sim_step_size_ms: int,
        max_sim_ms: int,
        device: str = "cpu",
        dtype: type[np.floating] = np.float64,
        logconfig: LogConfig | None = None,
        debug: bool | None = None,
        writer: SummaryWriter | None = None,
        **kwargs,  # TODO: 待定
    ):
        """
        向量化环境(类似 NVIDIA Isaac Gym)
        """
        self.__init_super(num_envs)
        num_envs = self.num_envs

        self._device = device  # np.device(device)
        self._dtype = _dtype = np.dtype(dtype).type
        self._writer = writer
        self._MASK1 = np.ones(
            (num_envs,),
            # device=device,
            dtype=np.bool_,
        )
        """环境掩码模板, shape=(N,)"""
        self.DEBUG = debug or self.__class__.DEBUG
        if logconfig:
            logr = logconfig.remake()
        else:
            logr = logging.getLogger(self.logger.name)
        self.logger = logr

        # time
        self._sim_step_size_ms = sim_step_size_ms
        assert max_sim_ms > 0 and np.isfinite(max_sim_ms), (
            "max_sim_ms must be positive finite number, got",
            max_sim_ms,
        )
        self.max_sim_time_ms = int(max_sim_ms)
        self._sim_time_ms = np.zeros(
            (num_envs, 1),
            # device=device,
            dtype=np.int64,
        )
        """仿真时间(ms)(用于精确事件控制), shape=(N, 1)"""
        self._sim_time_s = np.zeros(
            (num_envs, 1),
            # device=device,
            dtype=_dtype,
        )
        """仿真时间(s), shape=(N, 1)"""
        if len(kwargs):
            self.logger.warning(f"unused kwargs: {kwargs}")

    def __init_super(self, num_envs: int, copy: bool = True, **kwargs):
        super().__init__  # ref
        assert num_envs > 0, f"expect num_envs be positive, got {num_envs}"
        self._num_envs = num_envs
        self.is_vector_env = True
        self.copy = copy

        self.closed = False
        self.viewer: Any | None = None

        self._rewards = np.zeros((self.num_envs,), dtype=np.float64)
        self._terminateds = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncateds = np.zeros((self.num_envs,), dtype=np.bool_)
        self._actions = None

    @property
    def dtype(self) -> type[np.floating]:
        """浮点类型"""
        return self._dtype

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def proc_indices(self, env_indices: EnvIndexType | None, check=False):
        """对环境索引做预处理"""
        if env_indices is None:
            idx = slice(None)
        else:
            idx = env_indices

        if isinstance(idx, slice):
            idx = np.arange(
                self.num_envs,
                # device=self.device,
                dtype=np.intp,
            )[idx]
        elif isinstance(idx, ndarray):
            idx = idx.astype(np.intp)
        else:
            idx = np.asarray(
                idx,
                # device=self.device,
                dtype=np.intp,
            )
        # assert isinstance(env_indices, ndarray)
        if check:
            imax = idx.max()
            assert (
                imax < self.num_envs
            ), f"env_indices {idx} out of range [0, {self.num_envs})"
        return idx

    def proc_to_mask(self, mask: EnvMaskType | None):
        """
        调整到与 env_num 一致的 mask
        """
        tgt = self._MASK1  # ref
        if mask is None or mask is Ellipsis or mask is tgt:
            msk = tgt
        elif isinstance(mask, np.ndarray):
            assert mask.dtype == np.bool_, "mask must be bool type"
            if mask.shape == tgt.shape:
                msk = mask
            else:
                if mask.ndim == tgt.ndim:
                    msk = mask
                elif mask.ndim == tgt.ndim + 1:
                    msk = mask.squeeze(-1)
                else:
                    raise ValueError("mask shape mismatch",
                                     mask.shape, tgt.shape)

                msk = np.logical_and(
                    tgt, msk
                )  # mask&self._MASK1 做的是字节位运算,输入为int数组时会发生意料之外的结果!
            # msk = mask.to(self.device)
        elif mask == _SliceAll:
            msk = tgt
        elif isinstance(mask, (slice, list, tuple)):
            msk = np.zeros_like(tgt, dtype=np.bool_)
            msk[mask] = True
        else:
            raise TypeError("unsupported mask type", type(mask))
        assert msk.shape == tgt.shape, (
            "mask shape mismatch",
            msk.shape,
            tgt.shape,
        )
        return msk

    @property
    def sim_time_ms(self) -> Int_NDArr:
        """仿真时间(ms), shape=(N, 1)"""
        return self._sim_time_ms

    @property
    def sim_time_s(self) -> Float_NDArr:
        """仿真时间(s), shape=(N, 1)"""
        return self._sim_time_s

    def sync_sim_time(self, mask: EnvMaskType | None):
        """ms->s"""
        mask = self.proc_to_mask(mask)
        self._sim_time_s[mask] = self._sim_time_ms[mask] * 1e-3

    def reset(self, *, 
              seed: Optional[Union[int, List[int]]] = None,
              options: dict | None = None) -> tuple[NDArray, dict[str, Any]]:
        options = options or {}
        mask = options.get(self.OPTION_KEY_MASK, None)
        mask = self.proc_to_mask(mask)

        options[self.OPTION_KEY_MASK] = mask
        self.reset_async(seed=seed, options=options)
        rst = self.reset_wait(seed=seed, options=options)
        return rst

    def step(
        self, actions: NDArray
    ) -> tuple[NDArray, Float_NDArr, BoolNDArr, BoolNDArr, dict[str, Any]]:
        self.step_async(actions)
        rst = self.step_wait()
        return rst

    def reset_async(self, *, seed, options) -> None:
        """
        send reset signal to selected environments
        """
        self._check_running()

    def step_async(self, actions: NDArray) -> None:
        """
        send step signal to selected environments
        """
        self._check_running()
        self._actions = actions

    def step_wait(
        self,
    ) -> tuple[NDArray, Float_NDArr, BoolNDArr, BoolNDArr, NestedNPDict]:
        nenv = self.num_envs
        actions = self._actions
        assert (
            actions is not None
        ), "got None actions, step_async must be called before step_wait"
        assert len(actions) == nenv, (
            "actions mismatch",
            len(actions),
            nenv,
        )
        observations, rew, term, trunc, infos = self.step_core(actions)
        infos[self.INFO_KEY_FINAL_OBS] = observations
        infos[self.INFO_KEY_FINAL_INFO] = infos

        term = np.ravel(term)
        trunc = np.ravel(trunc)
        rew = np.ravel(rew)
        self._terminateds[:] = term
        self._truncateds[:] = trunc
        self._rewards[:] = rew

        done = np.ravel(term | trunc)
        if done.any():
            observations = deepcopy(observations)
            infos = deepcopy(infos)

            obs1, info1 = self.reset_core(done)
            observations[done] = obs1
            for k, dst in infos.items():
                if k not in info1:
                    continue
                elif isinstance(dst, dict):
                    continue
                elif isinstance(dst, ndarray):
                    dst[done] = info1[k]

        self.observations = observations
        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            np.copy(self._terminateds),
            np.copy(self._truncateds),
            infos,
        )

    @abstractmethod
    def reward_fns(self) -> list[BaseRewardFn]:
        """
        奖励函数组
        """
        pass

    def seeding(self, seed: int | np.ndarray | Sequence[int] | None = None) -> None:
        """
        重置随机种子(RNG 被全体实例共享)
        """
        if seed is not None:
            seed = int(np.ravel(seed)[0])
        self._np_random, seed = np_random(seed)

    def reset_sim_time(self, mask: EnvMaskType | None) -> None:
        """
        重置仿真时间
        """
        mask = self.proc_to_mask(mask)
        self._sim_time_ms[mask] = 0
        self.sync_sim_time(mask)

    @property
    @abstractmethod
    def single_observation_space(self) -> gymnasium.Space:
        """
        单例观测空间
        """
        pass

    @property
    @abstractmethod
    def single_action_space(self) -> gymnasium.Space:
        """
        单例动作空间
        """
        raise NotImplementedError

    @property
    def observation_space(self) -> gymnasium.Space:
        """
        环境组观测空间
        """
        raise NotImplementedError

    @property
    def action_space(self) -> gymnasium.Space:
        """
        环境组动作空间
        """
        raise NotImplementedError

    def _check_running(self):
        if self.closed:
            raise ValueError("Trying to operate on a closed environment")

    @abstractmethod
    def reset_core(self, mask: EnvMaskType | None) -> tuple[NDArray, dict[str, Any]]:
        r"""
        reset selected environments and return initial observations and infos from them.\
        NOTE: return in deep copy.\
        assert: return size == sum(mask).
        """
        raise NotImplementedError

    @abstractmethod
    def step_core(
        self, actions: NDArray
    ) -> tuple[NDArray, Float_NDArr, BoolNDArr, BoolNDArr, NestedNPDict]:
        r"""
        return next observations, rewards, terminals, truncateds, infos.\
        NOTE: 
            return in deep copy.\
            don't reset automatically when done.\
        """
        raise NotImplementedError

    def reset_wait(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ):
        r"""Waits for the calls triggered by :meth:`reset_async` to finish and returns the results.

        Args:
            seed: The reset environment seed, None->don't reset seed.
            options: Option information for the environment reset.

        Returns:
            The reset observation of the environment and reset information
            len(observations) == num_envs if `mask` is None or full mask else sum(`mask`).
        """
        nenv = self.num_envs
        if seed is None:
            # seed = [None for _ in range(self.num_envs)]
            pass
        else:
            self.seeding(seed)

        options = options or {}
        mask = options.get(self.OPTION_KEY_MASK, None)
        mask = self.proc_to_mask(mask)

        self._terminateds[mask] = False
        self._truncateds[mask] = False
        self._rewards[mask] = 0.0

        observations, infos = self.reset_core(mask)
        try:
            assert self.observations is not None, "self.observations is None"
        except:
            assert nenv == observations.shape[0], (
                f"expect len(observations)=={nenv} in first reset,got",
                observations.shape[0],
            )
            self.observations = observations
        self.observations[mask] = observations

        return (deepcopy(observations) if self.copy else observations), infos

    def close(self) -> None:
        if self._writer:
            self._writer.close()
