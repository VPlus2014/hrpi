from __future__ import annotations
from typing import Sequence, cast, TYPE_CHECKING

if TYPE_CHECKING:
    from ..proto4venv # import torchSyncVecEnv
from gymnasium import Wrapper, spaces
import numpy as np


class UnitActionWrapper(Wrapper):

    def __init__(self, env: TorchSyncVecEnv):
        """
        [-1,1]

        Args:
            env (Env): _description_

        """
        super().__init__(env)
        src_space = env.action_space
        assert isinstance(src_space, (spaces.Box,)), (
            "Unsupported action space type:",
            type(src_space),
        )
        np_float = cast(type[np.floating], src_space.dtype)
        assert np_float in (np.float32, np.float64), (
            "Unsupported float type:",
            np_float,
        )
        self._action_space = dst_space = spaces.Box(
            low=-1,
            high=1,
            shape=src_space.shape,
            dtype=np_float,
        )
        self._low = dst_space.low.astype(np_float)
        self._high = dst_space.high.astype(np_float)

    def step(self, action: np.ndarray) -> tuple:
        """
        Args:
            action: np.ndarray, shape= (...,dimA)
        """
        dimA = self._action_space.shape[0]
        rst = self.env.step(ac)
        return rst

    def _d2c(self, action: np.ndarray) -> np.ndarray:
        if not self._need_cvrt:
            return action
