import torch
from typing import TYPE_CHECKING
from .base_termination_fn import BaseTerminationFn

if TYPE_CHECKING:
    from environments.navigation import NavigationEnv

class TimeoutTerminationFn(BaseTerminationFn):
    def __init__(self, timeout_s: float) -> None:
        super().__init__()
        self.timeout_s = timeout_s

    def reset(self, **kwargs) -> None:
        pass

    def __call__(self, env: "NavigationEnv", **kwargs) -> torch.Tensor:
        _terminated = (env.sim_time_s >= self.timeout_s).detach()
        if torch.any(_terminated):
            pass
        return _terminated