from __future__ import annotations
from typing import Any

import numpy as np
from .proto4venv_wrapper import (
    VenvObservationWrapper,
    VectorEnv,
    WrapperObsType,
    ActType,
    ObsType,
)
from gymnasium import spaces
from ..utils.space_tf import affcmb_inv, batch_space


class ObsNormWrapper(
    VenvObservationWrapper[
        np.ndarray,
        ActType,
        np.ndarray,
    ]
):
    def __init__(self, env: VectorEnv):
        """bounded -> [0, 1]"""
        super().__init__(env)

        src_space = env.single_observation_space
        assert isinstance(src_space, spaces.Box), (
            "only support Box single observation space",
            type(src_space),
        )
        self._src_low = _src_low = src_space.low
        assert _src_low.ndim == 1, (
            "only support 1D single observation space",
            _src_low.shape,
        )
        self._src_span = _src_span = src_space.high - _src_low
        assert np.isfinite(self._src_low).all() and np.isfinite(self._src_span).all(), (
            "single observation space must be finite bounded",
            self._src_low,
            self._src_span,
        )
        assert _src_low.shape == _src_span.shape, (
            "low.shape!=high.shape",
            _src_low.shape,
            _src_span.shape,
        )
        self._single_obs_space = dst_space=spaces.Box(
            0, 1, shape=_src_low.shape, dtype=_src_low.dtype.type
        )
        self._obs_space = batch_space(dst_space, n=env.num_envs)

    def observation(self, observation: np.ndarray | Any) -> np.ndarray:
        obs = np.asarray(observation)
        low = self._src_low
        assert obs.ndim >= low.ndim
        shp2 = (1,) * (obs.ndim - low.ndim) + low.shape
        low = low.reshape(shp2)
        span = self._src_span.reshape(shp2)
        obs = affcmb_inv(obs, low, span)
        obs = obs.astype(low.dtype)
        return obs

    def reset_wait(self, **kwargs):
        rst = list(self.env.reset_wait(**kwargs))
        rst[0] = self.observation(rst[0])
        return tuple(rst)

    def step_wait(self, **kwargs):
        rst = list(self.env.step_wait(**kwargs))
        rst[0] = self.observation(rst[0])
        return tuple(rst)

    @property
    def single_observation_space(self) -> spaces.Box:
        return self._single_obs_space

    @property
    def observation_space(self) -> spaces.Box:
        return self._obs_space
