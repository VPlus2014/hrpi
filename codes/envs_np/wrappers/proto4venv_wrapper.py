from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, Union
from ..proto4venv import NPSyncVecEnv

# from gymnasium import Wrapper
from gymnasium.vector import VectorEnv, VectorEnvWrapper as _VectorEnvWrapper
from gymnasium import (
    spaces,
    ActionWrapper as _ActionWrapper,
    ObservationWrapper as _ObservationWrapper,
)

ObsType = TypeVar("ObsType", covariant=False)
ActType = TypeVar("ActType", covariant=False)
WrapperObsType = TypeVar("WrapperObsType", covariant=False)
WrapperActType = TypeVar("WrapperActType", covariant=False)


class VectorEnvWrapper(
    _VectorEnvWrapper,
    Generic[WrapperObsType, WrapperActType, ObsType, ActType],
    ABC,
):
    def __init__(self, env: VectorEnv):
        super().__init__(env)

    @property
    def single_observation_space(self) -> spaces.Space[WrapperObsType]:
        return self.env.single_observation_space

    @property
    def single_action_space(self) -> spaces.Space[WrapperActType]:
        return self.env.single_action_space

    @property
    def observation_space(self) -> spaces.Space[WrapperObsType]:
        return self.env.observation_space

    @property
    def action_space(self) -> spaces.Space[WrapperActType]:
        return self.env.action_space


class NPVenvWrapper(
    VectorEnvWrapper[WrapperObsType, WrapperActType, ObsType, ActType], ABC
):
    """(deprecated) NPVecEnv->gymnasium vector envs"""

    def __init__(self, env: NPSyncVecEnv):
        tgt = NPSyncVecEnv
        assert isinstance(env, tgt), (
            f"env must be an instance of {tgt}",
            type(env),
        )
        super().__init__(env)
        self.env = env

    def reset(self, *, seed=None, options: dict[str, Any] | None = None):
        options = options or {}
        return self.env.reset(seed=seed, options=options)

    def step(self, actions, **kwargs):
        self.step_async(actions, **kwargs)
        return self.step_wait()

    def step_async(self, actions):
        self._rst4step = self.env.step(actions)

    def step_wait(self):
        return self._rst4step

    def reset_async(self, seed=None, options: dict[str, Any] | None = None):
        options = options or {}
        self._rst4reset = self.env.reset(seed=seed, options=options)

    def reset_wait(self):
        return self._rst4reset


class VenvActionWrapper(VectorEnvWrapper[ObsType, WrapperActType, ObsType, ActType]):
    """
    see `gymnasium.ActionWrapper`
    """

    def __init__(self, env: VectorEnv):
        """Constructor for the action wrapper."""
        super().__init__(env)

    def step_async(self, actions: WrapperActType):
        actions_ = self.action(actions)
        return self.env.step_async(actions_)

    @abstractmethod
    def action(self, actions: WrapperActType) -> ActType:
        """Returns a modified action before :meth:`env.step` is called.

        Args:
            actions: The original :meth:`step` actions

        Returns:
            The modified actions
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def single_action_space(self) -> spaces.Space[WrapperActType]:
        """Returns the action space of a single environment."""
        raise NotImplementedError

    @property
    @abstractmethod
    def action_space(self) -> spaces.Space[WrapperActType]:
        """Returns the action space of the vector environment."""
        raise NotImplementedError


class VenvObservationWrapper(
    VectorEnvWrapper[WrapperObsType, ActType, ObsType, ActType]
):
    """
    see `gymnasium.ObsWrapper`
    """

    def __init__(self, env: VectorEnv):
        """Constructor for the observation wrapper."""
        super().__init__(env)

    def reset(self, **kwargs):
        return super().reset(**kwargs)

    @abstractmethod
    def observation(self, observation: ObsType) -> WrapperObsType:
        raise NotImplementedError

    @property
    @abstractmethod
    def observation_space(self) -> spaces.Space[WrapperObsType]:
        """Returns the observation space of the vector environment."""
        raise NotImplementedError

    @property
    @abstractmethod
    def single_observation_space(self) -> spaces.Space[WrapperObsType]:
        """Returns the observation space of a single environment."""
        raise NotImplementedError
