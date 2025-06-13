import torch
from typing import TYPE_CHECKING
from .base_termination_fn import BaseTerminationFn


class LowAltitudeTerminationFn(BaseTerminationFn):
    def __init__(self, min_altitude_m: float) -> None:
        super().__init__()
        self.min_altitude_m = min_altitude_m

    def reset(self, env, **kwargs) -> None:
        pass

    def forward(self, env, plane, **kwargs) -> torch.Tensor:
        return plane.altitude_m() < self.min_altitude_m
