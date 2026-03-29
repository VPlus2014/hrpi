from __future__ import annotations
from typing import Sequence
from gymnasium import ActionWrapper, spaces, Env

from gymnasium.vector import VectorEnv, VectorEnvWrapper
import numpy as np
from .proto4venv_wrapper import VenvActionWrapper, ObsType
from ..utils.space_tf import (
    discretize_space,
    undisc_with_gather,
    undisc_with_interp,
    batch_space,
)


class LinspaceActionWrapper(VenvActionWrapper[ObsType, np.ndarray, np.ndarray]):

    def __init__(self, env: VectorEnv, nvec: Sequence[int], use_table=False):
        """
        Args:
            env (VectorEnv):
            nvec (Sequence[int]): for each action dimension, the number of discrete values
            use_table (bool, optional): get action in searching table or interpolation(very fast). Defaults to False.
        """
        super().__init__(env)
        self._nvec = nvec = (*np.ravel(nvec),)
        self._dimA = len(nvec)
        assert all(n > 0 for n in nvec), (
            "expected nvec to be positive integers, got",
            nvec,
        )
        self._use_table = use_table

        src_space = (
            env.single_action_space if isinstance(env, VectorEnv) else env.action_space
        )
        dst_space, tables = discretize_space(src_space, nvec)
        self._single_action_space = dst_space
        self._action_space = batch_space(dst_space, n=env.num_envs)
        self._act_tables = tables
        self._act_low = np.ravel([t[0] for t in tables])
        self._act_high = np.ravel([t[-1] for t in tables])

    @property
    def single_action_space(self) -> spaces.MultiDiscrete:
        return self._single_action_space

    def action(self, actions: np.ndarray) -> np.ndarray:
        """
        Args:
            actions: np.ndarray, shape= (...,N,dimA)
        """
        assert len(actions.shape) >= 1, "Action should have at least 1 dimensions"
        dimA = self._dimA
        assert actions.shape[-1] == dimA, (
            f"expected action.shape[-1]=={dimA}, got",
            actions.shape[-1],
        )
        if self._use_table:  # speed: search < interpolation
            act_ = undisc_with_gather(actions, self._act_tables)
        else:
            act_ = undisc_with_interp(
                actions, self._nvec, self._act_low, self._act_high
            )
        return act_

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space


class FlattenMultiDiscreteActionWrapper(VenvActionWrapper):
    def __init__(self, env: VectorEnv):
        super().__init__(env)
        src_space = (
            env.single_action_space if isinstance(env, VectorEnv) else env.action_space
        )
        assert isinstance(src_space, spaces.MultiDiscrete), (
            "expected MultiDiscrete action space, got",
            type(env.action_space),
        )
        assert src_space.nvec.ndim == 1, (
            "expected nvec to be 1D tensor, got",
            src_space.nvec.ndim,
        )
        self._sizeA = np.prod(src_space.nvec)
        self._dimA = src_space.nvec.shape[0]
        self._nvec = src_space.nvec
        self._single_action_space = dst_space = spaces.Discrete(self._sizeA)
        self._action_space = batch_space(dst_space, n=env.num_envs)

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Args:
            action: np.ndarray, shape= (...,1)
        """
        nenv = self.num_envs
        if action.shape == (nenv, 1):
            pass
        elif action.shape == (nenv,):
            action = action.reshape(nenv, 1)
        else:
            raise ValueError(
                f"expected action.shape==({nenv},1) or ({nenv},), got {action.shape}"
            )
        assert len(action.shape) >= 1, "Action should have at least 1 dimensions"
        assert action.shape[-1] == 1, (
            "expected last dimension of action to be 1, got",
            action.shape[-1],
        )
        actnd = np.unravel_index(action, self._nvec)
        actnd = np.concatenate(actnd, axis=-1)  # (...,dimA)
        return actnd.astype(self.env.action_space.dtype)

    @property
    def single_action_space(self) -> spaces.Discrete:
        return self._single_action_space

    @property
    def action_space(self) -> spaces.MultiDiscrete:
        return self._action_space
