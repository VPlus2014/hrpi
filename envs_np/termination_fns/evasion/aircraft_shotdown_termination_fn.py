# import torch
from typing import TYPE_CHECKING
from ..base_termination_fn import BaseTerminationFn

if TYPE_CHECKING:
    from environments_th import EvasionEnv


class AircraftShotdownTerminationFn(BaseTerminationFn):
    def __init__(self) -> None:
        super().__init__()

    def reset(self, env: "EvasionEnv", **kwargs) -> None:
        pass

    def forward(self, env: "EvasionEnv", **kwargs) -> torch.Tensor:
        return env.aircraft.is_shotdown()
