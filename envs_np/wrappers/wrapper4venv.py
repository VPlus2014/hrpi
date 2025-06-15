from __future__ import annotations
from multiprocessing import RLock
from typing import Any, Callable, TYPE_CHECKING
import gymnasium

# import torch

if TYPE_CHECKING:
    from gymnasium.core import ObsType, ActType
    from ..proto4venv import *
import numpy as np


class SyncVEnvWrapper(gymnasium.Wrapper):
    # 并仿截断/终止处理机制, 参照 gymnasium.vector.sync_vector_env.SyncVectorEnv
    KEY_FINAL_OBS = "final_observation"  # 上一局终末状态
    KEY_FINAL_INFO = "final_info"  # 上一局调试信息
    KEY_FINAL_OBS_MASK = f"_{KEY_FINAL_OBS}"  # bool
    KEY_FINAL_INFO_MASK = f"_{KEY_FINAL_INFO}"

    def __init__(self, venv: SyncVecEnv, max_episode_steps: int | None = None):
        pass

    def reset(self, env_indices: EnvMaskType | None = None, **kwargs):
        pass

    def step(
        self, actions: np.ndarray, env_indices: EnvMaskType | None = None, **kwargs
    ):
        pass
