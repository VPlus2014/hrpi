from __future__ import annotations
from typing import Sequence
from gymnasium import Env, Wrapper, spaces
import numpy as np


def discretize_space(space: spaces.Box, nvec: Sequence[int]):
    if isinstance(space, spaces.Box):
        assert all(n > 0 for n in nvec), "nvec must be positive"
        return spaces.MultiDiscrete(np.array(nvec) - 1)
    else:
        raise TypeError("space must be Box, got", type(space))


class DiscActionWrapper(Wrapper):
    def __init__(self, env: Env, nvec: Sequence[int], flatten: bool = False):
        """
        连续动作空间分割

        Args:
            env (Env): _description_
            nvec (Sequence[int]): _description_
            flatten (bool, optional): _description_. Defaults to False.

        Raises:
            TypeError: _description_
        """
        super().__init__(env)
        self._nvec = nvec = (*nvec,)
        assert all(n > 0 for n in nvec), (
            "expected nvec to be positive integers, got",
            nvec,
        )
        src_act_space = env.action_space
        need_cvrt = False
        if isinstance(src_act_space, spaces.Box):
            need_cvrt = True
            dst_act_space = discretize_space(src_act_space, nvec)
        elif isinstance(src_act_space, (spaces.Discrete, spaces.MultiDiscrete)):
            pass
        else:
            raise TypeError(f"Unsupported action space type: {type(src_act_space)}")

        self._need_cvrt = need_cvrt

        assert (
            len(nvec) == self.env.action_space.shape[-1]
        ), "nvec should have the same length as the last dimension of the action space"
        act_is_disc = isinstance(env.action_space, spaces.Discrete)

    def step(self, action: np.ndarray) -> tuple:
        """
        Args:
            action: np.ndarray, shape= (...,N,dimA)
        """
        assert len(action.shape) >= 2, "Action should have at least 2 dimensions"
        ac = self._d2c(action)
        rst = self.env.step(ac)
        return rst

    def _d2c(self, action: np.ndarray) -> np.ndarray:
        if not self._need_cvrt:
            return action
