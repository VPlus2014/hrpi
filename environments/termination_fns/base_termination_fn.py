import torch
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
if TYPE_CHECKING:
    from environments.navigation import NavigationEnv

class BaseTerminationFn(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def reset(self, **kwargs) -> None:
        ...

    @abstractmethod
    def __call__(self, env: "NavigationEnv", **kwargs) -> torch.Tensor:
        ...