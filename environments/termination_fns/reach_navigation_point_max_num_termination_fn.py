import torch
from typing import TYPE_CHECKING
from .base_termination_fn import BaseTerminationFn

if TYPE_CHECKING:
    from environments.navigation import NavigationEnv


class ReachNavigationPointMaxNumTerminationFn(BaseTerminationFn):
    def __init__(self) -> None:
        super().__init__()

    def reset(self, **kwargs) -> None:
        pass

    def __call__(self, env: "NavigationEnv", **kwargs) -> torch.Tensor:
        return (env.navigation_point_index[:, :, 0] >= env.navigation_points_total_num).detach()