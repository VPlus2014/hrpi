from __future__ import annotations
from typing import Sequence
from gymnasium import ActionWrapper, Env, spaces
import numpy as np


def discretize_space(
    space: spaces.Space,
    nvec: Sequence[int] | np.ndarray,
    dtype: type[np.integer] = np.intp,
) -> tuple[spaces.MultiDiscrete, list[np.ndarray]]:
    assert all(n > 0 for n in nvec), ("nvec must be positive", nvec)
    dimAd = len(nvec)
    space_d = spaces.MultiDiscrete(np.array(nvec, dtype=dtype) - 1, dtype=dtype)
    if isinstance(space, spaces.Box):
        src_ndim = len(space.shape)
        assert src_ndim == 1, ("Box space must be 1-dimensional", src_ndim)
        dimA0 = space.shape[0]
        assert dimAd == dimA0, ("expected len(nvec)==space.shape[0], got", dimAd, dimA0)
        tables = [
            np.linspace(space.low[i], space.high[i], nvec[i], dtype=space.dtype)
            for i in range(dimAd)
        ]
    elif isinstance(space, spaces.Discrete):
        assert dimAd == 1, ("expected len(nvec)==1, got", dimAd)
        tables = [
            np.linspace(
                space.start,
                space.start + space.n - 1,
                nvec[0],
                dtype=space.dtype,
            )
        ]
    elif isinstance(space, spaces.MultiDiscrete):
        src_ndim = len(space.nvec)
        assert dimAd == src_ndim, (
            "expected len(nvec)==len(space.nvec), got",
            dimAd,
            src_ndim,
        )
        tables = [
            np.linspace(0, space.nvec[i], nvec[i], dtype=space.dtype)
            for i in range(dimAd)
        ]
    else:
        raise TypeError("space must be Box, got", type(space))
    return space_d, tables


class DiscActionWrapper(ActionWrapper):
    def __init__(self, env: Env, nvec: Sequence[int]):
        """
        连续动作空间均等分割

        Args:
            env (Env):
            nvec (Sequence[int]):
        """
        super().__init__(env)
        self._nvec = nvec = (*nvec,)
        self._dimA = len(nvec)
        assert all(n > 0 for n in nvec), (
            "expected nvec to be positive integers, got",
            nvec,
        )
        src_act_space = env.action_space
        dst_act_space, tables = discretize_space(src_act_space, nvec)
        self.action_space = dst_act_space
        self._act_tables = tables

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Args:
            action: np.ndarray, shape= (...,N,dimA)
        """
        assert len(action.shape) >= 1, "Action should have at least 1 dimensions"
        dimA = self._dimA
        assert action.shape[-1] == dimA, (
            f"expected last dimension of action to be {dimA}, got",
            action.shape[-1],
        )
        # 逐元素查找离散化表
        ndim = len(action.shape)
        _1s = (1,) * (ndim - 1)
        tab = self._act_tables
        act_ = [
            np.take_along_axis(
                tab[i].reshape(_1s + (-1,)),
                action[..., i : i + 1],
                axis=-1,
            )
            for i in range(dimA)
        ]
        act_ = np.concatenate(act_, axis=-1)
        return act_
