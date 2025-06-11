from __future__ import annotations
from typing import TYPE_CHECKING, Union, Sequence

if TYPE_CHECKING:
    from .simulators.base_model import BaseModel
import gymnasium
import torch
from gymnasium.vector.async_vector_env import AsyncVectorEnv

_EnvIndexType = Union[torch.Tensor, Sequence[int], slice, None]

# from gymnasium.vector.sync_vector_env import SyncVectorEnv
_DeviceLikeType = Union[torch.device, str, int]


class CUDAVecEnv(gymnasium.Env):
    name: str

    # 并仿截断/终止处理机制, 参照 gymnasium.vector.sync_vector_env.SyncVectorEnv
    KEY_FINAL_OBS = "final_observation"  # 上一局终末状态
    KEY_FINAL_INFO = "final_info"  # 上一局调试信息
    KEY_FINAL_OBS_MASK = f"_{KEY_FINAL_OBS}"  # bool
    KEY_FINAL_INFO_MASK = f"_{KEY_FINAL_INFO}"

    def __init__(
        self,
        num_envs: int,
        device: _DeviceLikeType,
        dtype: torch.dtype,
        **kwargs,  # TODO: 待定
    ):
        """
        张量化环境(类似 NVIDIA Isaac Gym)\
        约定:
            当发生截断/终止时, step 返回的
        """
        self._device = torch.device(device)
        self._dtype = dtype
        assert dtype in (
            torch.float32,
            torch.float64,
        ), (
            f"unsupported torch float dtype {dtype}",
            "only float32 and float64 are supported",
        )
        assert num_envs > 0, f"num_envs must be positive, got {num_envs}"
        self._num_envs = num_envs

        self._sim_time_ms = torch.zeros(
            (num_envs, 1), device=device, dtype=torch.int64
        )  # 仿真时间(ms)(用于精确事件控制)
        self._sim_time_s = torch.zeros(
            (num_envs, 1), device=device, dtype=dtype
        )  # 仿真时间(s)

    @property
    def device(self) -> torch.device:
        """仿真设备"""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """torch浮点类型"""
        return self._dtype

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def proc_indices(self, env_indices: _EnvIndexType = None, check=False):
        """对环境索引做预处理"""
        if env_indices is None:
            idx = slice(None)
        else:
            idx = env_indices

        if isinstance(idx, slice):
            idx = torch.arange(self.num_envs, device=self.device, dtype=torch.int64)[
                idx
            ]
        elif isinstance(idx, torch.Tensor):
            pass
        else:
            idx = torch.asarray(idx, device=self.device, dtype=torch.int64)
        # assert isinstance(env_indices, torch.Tensor)
        if check:
            imax = idx.max().item()
            assert (
                imax < self.num_envs
            ), f"env_indices {idx} out of range [0, {self.num_envs})"
        return idx

    @property
    def sim_time_ms(self) -> torch.Tensor:
        """仿真时间(ms), shape=(N, 1)"""
        return self._sim_time_ms

    @property
    def sim_time_s(self) -> torch.Tensor:
        """仿真时间(s), shape=(N, 1)"""
        return self._sim_time_s

    def sync_sim_time(self, index=None):
        """ms->s"""
        index = self.proc_indices(index)
        self._sim_time_s[index] = self._sim_time_ms[index] * 1e-3
