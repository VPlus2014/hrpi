from __future__ import annotations
from typing import Sequence, cast, TYPE_CHECKING

from gymnasium import Wrapper, spaces
import numpy as np

if TYPE_CHECKING:
    from gymnasium import Env


class UnitActionWrapper(Wrapper):

    def __init__(self, env: Env):
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
        np_float = cast(type[np.floating], np.dtype(src_space.dtype).type)
        assert np_float in (np.float32, np.float64), (
            "Unsupported float type:",
            np_float,
        )

        self.action_space = dst_space = spaces.Box(
            low=-1,
            high=1,
            shape=src_space.shape,
            dtype=np_float,
        )
        self._low = dst_space.low.astype(np_float)
        self._high = dst_space.high.astype(np_float)
        self._span = self._high - self._low

    def step(self, action: np.ndarray) -> tuple:
        """
        Args:
            action: np.ndarray, shape= (...,dimA)
        """
        act_ = ((action + 1) * 0.5) * self._span + self._low
        rst = self.env.step(act_)
        return rst
