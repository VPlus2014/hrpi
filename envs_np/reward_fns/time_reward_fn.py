from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..proto4venv # import torchSyncVecEnv
    from .base_reward_fn import _EnvIndexType
# import torch
import math
from .base_reward_fn import BaseRewardFn


class TimeRewardFn(BaseRewardFn):
    def __init__(self, weight: float = 1) -> None:
        super().__init__()
        self.weight = weight

    def reset(self, env: TorchSyncVecEnv, env_indices: _EnvIndexType = None):
        pass

    def forward(self, env: TorchSyncVecEnv, **kwargs) -> torch.Tensor | float:
        # return self.weight * torch.ones(
        #     (env.num_envs, 1), device=env.device, dtype=env.dtype
        # )
        return 1.0
