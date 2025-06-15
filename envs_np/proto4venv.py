from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Union, Sequence


import gymnasium
from gymnasium.utils.seeding import np_random

from numpy import ndarray
import numpy as np
from numpy.typing import NDArray
import logging

if TYPE_CHECKING:
    from .simulators.proto4model import BaseModel
    from .utils.math_np import Float_NDArr, Int_NDArr, BoolNDArr
    from .utils.log_ext import LogConfig
    from .reward_fns.proto4rf import BaseRewardFn
EnvIndexType = Union[ndarray, NDArray[np.intp], list[int], tuple[int], slice]
EnvMaskType = Union[
    ndarray, NDArray[np.bool_], list[bool], tuple[bool], type(Ellipsis), slice
]
# from gymnasium.vector.sync_vector_env import SyncVectorEnv
_SliceAll = slice(None)


class SyncVecEnv(gymnasium.Env[NDArray, NDArray], ABC):

    DEBUG: bool = False
    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(
        self,
        num_envs: int,
        sim_step_size_ms: int,
        max_sim_ms: int,
        device: str = "cpu",
        dtype: type[np.floating] = np.float64,
        logconfig: LogConfig | None = None,
        debug: bool | None = None,
        **kwargs,  # TODO: 待定
    ):
        """
        向量化环境(类似 NVIDIA Isaac Gym)
        """
        self._device = device  # np.device(device)
        self._dtype = _dtype = np.dtype(dtype).type
        assert num_envs > 0, f"num_envs must be positive, got {num_envs}"
        self._num_envs = num_envs
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
            logr = self.logger
        self.logger = logr

        # time
        self._sim_step_size_ms = sim_step_size_ms
        assert max_sim_ms > 0, ("max_sim_ms must be positive", max_sim_ms)
        self.max_sim_time_ms = max_sim_ms
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
                    raise ValueError("mask shape mismatch", mask.shape, tgt.shape)

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

    @abstractmethod
    def reset(
        self, mask: EnvMaskType | None = None, **kwargs
    ) -> tuple[NDArray, dict[str, Any]]:
        return super().reset(**kwargs)

    @abstractmethod
    def step(
        self, action: NDArray, mask: EnvMaskType | None = None, **kwargs
    ) -> tuple[NDArray, Float_NDArr, BoolNDArr, BoolNDArr, dict[str, Any]]:
        pass

    @abstractmethod
    def reward_fns(self) -> list[BaseRewardFn]:
        """
        奖励函数组
        """
        pass

    def seeding(self, seed: int | None = None) -> None:
        """
        重置随机种子
        """
        self._np_random, seed = np_random(seed)
