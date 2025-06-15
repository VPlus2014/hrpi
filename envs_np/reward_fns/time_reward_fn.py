from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..proto4venv import SyncVecEnv
    from .proto4rf import EnvMaskType
# import torch
import math
from .proto4rf import BaseRewardFn, RewardType


class RF_TimeCost(BaseRewardFn):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def reset(self, env: SyncVecEnv, env_indices: EnvMaskType | None = None, **kwargs):
        pass

    def forward(self, env: SyncVecEnv, **kwargs) -> RewardType:
        # return self.weight * np.ones(
        #     (env.num_envs, 1), device=env.device, dtype=env.dtype
        # )
        return -1.0
