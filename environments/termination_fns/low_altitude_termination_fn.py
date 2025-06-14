import torch
from typing import TYPE_CHECKING
from .base_termination_fn import BaseTerminationFn

if TYPE_CHECKING:
    from environments.navigation import NavigationEnv

class LowAltitudeTerminationFn(BaseTerminationFn):
    def __init__(self, min_altitude_m: float) -> None:
        super().__init__()
        self.min_altitude_m = min_altitude_m

    def reset(self, **kwargs) -> None:
        pass

    def __call__(self, env: "NavigationEnv", **kwargs) -> torch.Tensor:
        return (env.aircraft.altitude_m < self.min_altitude_m).detach()