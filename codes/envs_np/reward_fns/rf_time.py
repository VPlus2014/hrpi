from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..proto4venv import NPSyncVecEnv
    from .proto4rf import EnvMaskType
# import torch
import math
from .proto4rf import BaseRewardFn, RewardType


class RF_TimeCost(BaseRewardFn):
    """耗时代价"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def reset(self, env: NPSyncVecEnv, env_indices: EnvMaskType | None = None, **kwargs):
        pass

    def forward(self, env: NPSyncVecEnv, unit, **kwargs) -> RewardType:
        return -1.0
    
class RF_TimeAlive(BaseRewardFn):
    """生存时间收益"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def reset(self, env: NPSyncVecEnv, env_indices: EnvMaskType | None = None, **kwargs):
        pass

    def forward(self, env: NPSyncVecEnv, unit, **kwargs) -> RewardType:
        return 1.0
