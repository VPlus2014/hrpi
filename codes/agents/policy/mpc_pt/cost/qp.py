from __future__ import annotations
from .proto4cost import BatchCostFn
import torch
from .utils import shape_rjust


class BatchQPCostFn(BatchCostFn):
    r"""
    向量化二次代价函数, 终末代价采用 $\hat{V}\leq V_\infty$ 型
    """

    def __init__(self, Q: torch.Tensor, R: torch.Tensor, Qf: torch.Tensor):
        """
        Args:
            Q: 状态权重矩阵 [dimX, dimX]
            R: 控制权重矩阵 [dimU, dimU]
            Qf: 终端状态权重矩阵 [dimX, dimX]
        """
        self.Q = Q
        self.R = R
        self.Qf = Qf

    def stage_cost(
        self,
        state: torch.Tensor,
        control: torch.Tensor,
        state_ref: torch.Tensor,
        control_ref: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        向量化阶段代价计算
        Args:
            state: (..., dimX)
            control: (..., dimU)
            state_ref: (..., dimX)
            control_ref: (..., dimU) 或 None
        Returns:
            cost: (...,1) 单步代价
        """

        # 向量化二次型计算: (x-x_ref)^T Q (x-x_ref)
        state_ref = shape_rjust(state_ref, state)
        state_error = state - state_ref
        Q = shape_rjust(self.Q, state_error)
        state_cost = (state_error * (state_error @ Q)).sum(dim=-1, keepdim=True)

        if control_ref is not None:
            control_error = control - control_ref
        else:
            control_error = control

        R = shape_rjust(self.R, control_error)
        control_cost = (control_error * (control_error @ R)).sum(dim=-1, keepdim=True)
        return state_cost + control_cost

    def terminal_cost(
        self, state: torch.Tensor, state_ref: torch.Tensor
    ) -> torch.Tensor:
        """
        向量化终端代价计算(启发式估计)
        Args:
            state: 状态 shape=(..., dimX)
            state_ref: 参考状态 shape=(..., dimX)
        Returns:
            cost: 终端代价 shape=(..., 1)
        """
        state_ref = shape_rjust(state_ref, state)
        state_error = state - state_ref
        Qf = shape_rjust(self.Qf, state_error)
        vf = (state_error * (state_error @ Qf)).sum(dim=-1, keepdim=True)
        return vf
