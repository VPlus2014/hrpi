# import torch
from typing import TYPE_CHECKING
from ..base_termination_fn import BaseTerminationFn

if TYPE_CHECKING:
    from ...navigation import NavigationEnv


class TC_FarAwayFromWaypoint(BaseTerminationFn):
    def __init__(self, distance_threshold: float) -> None:
        """距离过远"""
        super().__init__()
        self.distance_threshold = distance_threshold

    def reset(self, env: "NavigationEnv", **kwargs) -> None:
        pass

    def forward(self, env: "NavigationEnv", plane, **kwargs) -> torch.Tensor:
        return env.cur_nav_dist >= self.distance_threshold
