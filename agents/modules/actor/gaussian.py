from __future__ import annotations
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Sequence

import torch
from torch import nn
from torch.distributions import Normal

_DEBUG = True


class GaussianActor(nn.Module):
    """连续有界高斯策略"""

    logr = logging.getLogger(__name__)

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_min: torch.Tensor,
        action_max: torch.Tensor,
        hidden_sizes: Sequence[int] = (),
        logr: logging.Logger = logr,
        max_std_ratio: float = 0.1,
    ):
        super().__init__()
        self.logr = logr
        logr.debug(f"action_min: {action_min}, action_max: {action_max}")
        action_span = action_max - action_min
        self.register_buffer("action_min", action_min)
        self.register_buffer("action_max", action_max)
        self.register_buffer("action_span", action_span)
        self.action_min = self.get_buffer("action_min")
        self.action_max = self.get_buffer("action_max")
        self.action_span = self.get_buffer("action_span")
        self.max_std_ratio = max_std_ratio

        from ..utils import MLP, init_net

        self._net = MLP(state_dim, action_dim + action_dim, hidden_sizes)

        init_net(self._net, init_type="orthogonal", init_gain=0.01)

    def get_kern(self, state: torch.Tensor):
        r"""获取[0,1]上的高斯分布参数(\mu, \sigma)"""
        mu, std = torch.chunk(self._net(state), 2, -1)
        mu = torch.sigmoid(mu)  # Ensure the output is in [0, 1]
        std = torch.sigmoid(std) * (
            1 / 6 * self.max_std_ratio
        )  # Ensure the std is in [0, 1/6*a]
        if _DEBUG:
            self.logr.debug(
                {
                    "mu_k": mu.mean().item(),
                    "std_k": std.mean().item(),
                }
            )

        return mu, std

    def forward(self, state: torch.Tensor):
        mu, std = self.get_kern(state)
        action = mu * self.action_span + self.action_min
        return action

    def get_dist(self, state: torch.Tensor) -> Normal:
        if _DEBUG:
            xinf = torch.isinf(state)
            xnan = torch.isnan(state)
            wher = torch.where(xinf | xnan)[0]
            self.logr.debug(
                {
                    "xinf": xinf.any().item(),
                    "xnan": xnan.any().item(),
                    # "where(xinf|xnan)": wher.cpu().numpy().tolist(),
                    # "x": state[wher].cpu().numpy().tolist(),
                }
            )
        mu_k, std_k = self.get_kern(state)
        dist = Normal(mu_k, std_k)
        return dist
