from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..navigation import NavigationEnv
    from ..evasion import EvasionEnv

import torch
from abc import ABC, abstractmethod

from typing import Union, Sequence, TypeVar

IndexLike = TypeVar("IndexLike", Sequence[int], torch.Tensor)


class BaseRewardFn(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def reset(
        self,
        env: NavigationEnv | EvasionEnv,
        env_indices: IndexLike | None = None,
        **kwargs,
    ) -> None: ...

    @abstractmethod
    def __call__(
        self, env: NavigationEnv | EvasionEnv, **kwargs
    ) -> torch.Tensor | float: ...
