from __future__ import annotations

import torch
from .proto4model import BatchDynamics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .utils import BatchODESolverType


class BatchCarModel(BatchDynamics):
    def __init__(self, dt: float, solver: BatchODESolverType):
        self.solver = solver
        super().__init__(5, 2, dt)

    def _dynamics(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        assert state.shape[-1] == self.dimX
        assert control.shape[-1] == self.dimU
        x, y, V, c, s = torch.split(state, 1, dim=-1)
        udV, udtheta = torch.split(control, 1, dim=-1)
        dVmax = 9.8
        dtheta_max = 2 * torch.pi / 4
        dtheta = udtheta * dtheta_max
        dotx = V * c
        doty = V * s
        dotV = udV * dVmax
        dotc = -s * dtheta
        dots = c * dtheta
        dotX = torch.cat([dotx, doty, dotV, dotc, dots], dim=-1)
        return dotX

    def forward(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        dt = self.dt
        f = self._dynamics
        y = self.solver(f, state, control, dt)
        return y
