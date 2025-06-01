from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Sequence
import gymnasium
import torch


class TrueVecEnv(gymnasium.Env):
    name: str

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        dtype: torch.dtype,
        **kwargs,
    ):
        """张量化环境"""
        self._device = device
        self._dtype = dtype
        self._num_envs = num_envs

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

    def proc_indices(
        self, indices: Sequence[int] | torch.Tensor | None = None, check=False
    ):
        """对索引做预处理"""
        num_envs = self.num_envs
        if indices is None:
            indices = torch.arange(num_envs, device=self.device)
        elif isinstance(indices, torch.Tensor):
            pass
        else:
            indices = torch.asarray(indices, device=self.device, dtype=torch.int64)
        # assert isinstance(env_indices, torch.Tensor)
        if check:
            imax = indices.max().item()
            assert (
                imax < self.num_envs
            ), f"env_indices {indices} out of range [0, {self.num_envs})"
        return indices
