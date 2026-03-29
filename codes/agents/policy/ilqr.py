from __future__ import annotations

from typing import Union
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from .utils.net import NNModule

DeviceLike = Union[torch.device, str, int]
TensorLike = Union[np.ndarray, torch.Tensor]


class iLQR(NNModule):

    DEBUG = False

    def __init__(
        self,
        model: nn.Module,
        horizon: int = 10,  # horizon
        lr=0.1,
        device: DeviceLike = "cpu",
        dtype=torch.float32,
    ):
        super().__init__()
        self.T = horizon = int(horizon)
        assert horizon > 0, ("horizon must be a positive integer", horizon)
        self.model = model
        self.state_dim = dimX = int(model.num_states)
        self.action_dim = dimU = int(model.num_actions)

        assert dtype in [
            torch.float32,
            torch.float64,
        ], ("dtype must be float32 or float64", dtype)

        self.lr = lr
        self.K = nn.Parameter(torch.randn(horizon, dimU, dimX))  # (N,dimU, dimX)
        self.b = nn.Parameter(torch.randn(horizon, dimU, 1))  # (N,dimU)
        self.xbar = nn.Parameter(torch.randn(horizon, dimX, 1))  # (N,dimX)

        self.optim = optim.Adam([self.K, self.b, self.xbar], self.lr, weight_decay=1e-6)

        self.to(device=device, dtype=dtype)

    @torch.no_grad()
    def reset(self):
        self.K.zero_()
        self.b.zero_()
        self.xbar.zero_()

    def forward(self, state: TensorLike):
        x = torch.as_tensor(state, device=self.device, dtype=self.dtype)  # (...,dimX)
        assert (
            x.shape[-1] == self.state_dim
        ), f"expected input shape (...,{self.state_dim}) but got {x.shape[-1]}"

    def shift(self):
        """receeding horizon"""
        pass

    def update(self, state, epochs=2):
        model = self.model
        device = self.device

        for epoch in range(epochs):
            s = torch.FloatTensor(state).unsqueeze(0).to(device)
            cost = 0.0
            for K, k, xbar in zip(self.K, self.b, self.xbar):
                u = torch.mv(K, xbar - s.squeeze()) + k
                s, r = model.step(s, u.unsqueeze(0))
                cost = cost - r
            if isinstance(cost, torch.Tensor):
                self.optim.zero_grad()
                cost.backward()
                self.optim.step()
            else:
                raise TypeError("cost must be a tensor", type(cost))

        with torch.no_grad():
            K = self.K[0].cpu().clone().numpy()
            k = self.b[0].cpu().clone().numpy()
            xbar = self.xbar[0].cpu().clone().numpy()
            self.b[:-1] = self.b[1:].clone()
            self.b[-1].zero_()
            self.K[:-1] = self.K[1:].clone()
            self.K[-1].zero_()
            self.xbar[:-1] = self.xbar[1:].clone()
            self.xbar[-1].zero_()
            return np.dot(K, xbar - state) + k
