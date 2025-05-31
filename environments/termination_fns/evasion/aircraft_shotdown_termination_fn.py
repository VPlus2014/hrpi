import torch
from typing import TYPE_CHECKING
from ..base_termination_fn import BaseTerminationFn

if TYPE_CHECKING:
    from environments import EvasionEnv

class AircraftShotdownTerminationFn(BaseTerminationFn):
    def __init__(self) -> None:
        super().__init__()

    def reset(self, **kwargs) -> None:
        pass

    def __call__(self, env: "EvasionEnv", **kwargs) -> torch.Tensor:
        return env.aircraft.is_shotdown()