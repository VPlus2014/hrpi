import torch
from typing import TYPE_CHECKING, Sequence
from abc import ABC, abstractmethod
if TYPE_CHECKING:
    from environments.navigation import NavigationEnv


class BaseRewardFn(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def reset(self, env: "NavigationEnv", env_indices: Sequence[int] | torch.Tensor | None = None, **kwargs) -> None:
        ...

    @abstractmethod
    def __call__(self, env: "NavigationEnv", **kwargs) -> torch.Tensor:
        ...