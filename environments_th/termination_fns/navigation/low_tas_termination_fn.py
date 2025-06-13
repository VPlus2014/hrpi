import torch
from typing import TYPE_CHECKING
from ..base_termination_fn import BaseTerminationFn

if TYPE_CHECKING:
    from ...navigation import NavigationEnv


class LowAirSpeedTerminationFn(BaseTerminationFn):

    def __init__(self, min_airspeed_mps: float) -> None:
        super().__init__()
        self.min_airspeed_mps = min_airspeed_mps

    def reset(self, env: "NavigationEnv", **kwargs) -> None:
        pass

    def forward(self, env: "NavigationEnv", plane, **kwargs) -> torch.Tensor:
        return env.aircraft.tas() < self.min_airspeed_mps
