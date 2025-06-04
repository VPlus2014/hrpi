import torch
from typing import TYPE_CHECKING
from ..base_termination_fn import BaseTerminationFn

if TYPE_CHECKING:
    from environments import EvasionEnv

class AircraftSurvivalTerminationFn(BaseTerminationFn):
    def __init__(self) -> None:
        super().__init__()

    def reset(self,env: "EvasionEnv", **kwargs) -> None:
        pass

    def forward(self, env: "EvasionEnv", **kwargs) -> torch.Tensor:
        return env.missile.is_missed()