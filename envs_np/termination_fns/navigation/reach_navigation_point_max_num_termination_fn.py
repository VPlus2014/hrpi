# import torch
from typing import TYPE_CHECKING
from ..base_termination_fn import BaseTerminationFn

if TYPE_CHECKING:
    from ...navigation import NavigationEnv


class ReachNavigationPointMaxNumTerminationFn(BaseTerminationFn):
    def __init__(self) -> None:
        super().__init__()

    def reset(self, env: "NavigationEnv", **kwargs) -> None:
        pass

    def forward(self, env: "NavigationEnv", plane, **kwargs) -> torch.Tensor:
        return env.cur_nav_point_index[:, 0, 0:1] >= env.waypoints_total_num
        # (B,1)
