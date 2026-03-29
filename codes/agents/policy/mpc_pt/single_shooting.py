# single shooting MPC
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import traceback
from matplotlib import pyplot as plt
from matplotlib import patches as patches, axes
import torch
import torch.nn as nn
import numpy as np
from typing import (
    Optional,
    Sequence,
    Tuple,
    Callable,
    Dict,
    Any,
    List,
    TypeVar,
    TYPE_CHECKING,
)
import warnings

from util_tools import shape_rjust, fit_lim1d
from .dynamics.utils import *
from .dynamics.proto4model import BatchDynamics

if TYPE_CHECKING:
    from .cost.proto4cost import BatchCostFn

AutoGradFunc = Callable[[torch.Tensor, torch.Tensor, Any], torch.Tensor]


class BatchMPCSolver:
    """
    向量化并行 MPC 无约束优化器
    """

    def __init__(
        self,
        dynamics_model: BatchDynamics,
        cost_function: BatchCostFn,
        horizon: int,
        control_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
        single_shooting: bool = True,
        leq_constraints: Sequence[AutoGradFunc] = (),
        gamma: float | Sequence[float] = 0.99,
        lr: float = 1.0,
        max_iter: int = 20,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        history_size: int = 100,
        use_line_search: bool = False,
    ):
        """
        Args:
            dynamics_model: 向量化离散时间动力学模型
            cost_function: 向量化代价函数
            horizon: 预测视界
            control_bounds: 控制约束 (lower, upper)
            batch_size: 并行优化的批次大小
            device: 计算设备
            gamma: 代价折扣因子
        """
        self.dynamics_model = dynamics_model
        self.cost_function = cost_function
        self.horizon = horizon
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        assert dtype in (torch.float32, torch.float64), (
            "dtype should be float32 or float64",
            dtype,
        )
        self.single_shooting = single_shooting
        assert single_shooting, NotImplementedError("multi-shooting not implemented")

        self.dimX = dimX = dynamics_model.dimX
        self.dimU = dimU = dynamics_model.dimU

        self.gamma = _gamma = torch.asarray(gamma, dtype=dtype, device=device).view(
            -1, 1
        )
        assert ((_gamma >= 0) & (_gamma <= 1)).all(), (
            "gamma should be in [0, 1]",
            gamma,
        )
        self._gammas = (
            gamma ** torch.arange(horizon + 1, device=device).reshape(-1, 1)
        ).type(dtype)

        self._Ubar = nn.Parameter(
            torch.zeros((batch_size, horizon, dimU), dtype=dtype, device=device)
        )
        self._Xbar = nn.Parameter(
            torch.zeros(batch_size, horizon + 1, dimX, device=device),
            requires_grad=not single_shooting,
        )
        self._X0 = torch.zeros(batch_size, dimX, device=device)
        self._Xbar_d = torch.zeros(batch_size, horizon + 1, dimX, device=device)

        if control_bounds is not None:
            assert all(isinstance(el, torch.Tensor) for el in control_bounds), (
                "control_bounds should be a tuple of two tensors",
                control_bounds,
            )
            ulb, uub = control_bounds
            assert (ulb <= uub).all(), (
                "expecting U lower bound <= U upper bound",
                ulb,
                uub,
            )
            self.control_bounds = (
                shape_rjust(ulb.to(device), self._Ubar),
                shape_rjust(uub.to(device), self._Ubar),
            )
            self._project_ubar()
        else:
            self.control_bounds = None

        # if state_bounds is not None:
        #     assert all(isinstance(el, torch.Tensor) for el in state_bounds), (
        #         "state_bounds should be a tuple of two tensors",
        #         state_bounds,
        #     )
        #     xlb, xub = state_bounds
        #     assert (xlb <= xub).all(), (
        #         "expecting X lower bound <= X upper bound",
        #         xlb,
        #         xub,
        #     )
        #     self.state_bounds = (
        #         shape_rjust(xlb.to(device), self._Xbar),
        #         shape_rjust(xub.to(device), self._Xbar),
        #     )
        # else:
        #     self.state_bounds = None

        # 优化器参数
        self.optimizer_params = {
            "lr": lr,
            "max_iter": max_iter,
            "tolerance_grad": tolerance_grad,
            "tolerance_change": tolerance_change,
            "history_size": history_size,
            "line_search_fn": "strong_wolfe" if use_line_search else None,
        }
        # 初始化优化器
        self._optm = self._make_solver()
        self._reuse_optm = True

        return

    def _make_solver(self):
        optr = torch.optim.LBFGS(self._solver_parameters(), **self.optimizer_params)
        return optr

    def _make_lam(self, constraints: Sequence[AutoGradFunc]):
        raise NotImplementedError

    def _make_constraints(self):
        if self.control_bounds is not None:
            ulb, uub = self.control_bounds
            gulb = ulb - self._Ubar
            guub = self._Ubar - uub

    def _predict_trajectory(
        self, initial_states: torch.Tensor, control_sequence: torch.Tensor
    ) -> torch.Tensor:
        """
        轨迹预测(single shooting)
        Args:
            initial_states: 初始状态, shape=(..., dimX)
            control_sequence: 控制序列,shape=(..., N, dimU)
        Returns:
            state_trajectory: (..., N+1, dimX) 状态轨迹
        """
        fd = self.dynamics_model
        states = [initial_states]  # 初始状态
        current_state = initial_states
        for t in range(self.horizon):
            control = control_sequence[..., t, :]
            next_state = fd(current_state, control)
            assert next_state.shape == current_state.shape
            states.append(next_state)
            current_state = next_state

        xbar = torch.stack(states, dim=-2)  # (..., N+1, dimX)
        return xbar

    def _compute_cost(
        self,
        x0: torch.Tensor,
        xbar: torch.Tensor,
        ubar: torch.Tensor,
        xbar_refs: torch.Tensor,
        ubar_refs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        向量化代价计算
        Args:
            x0: 初始状态 shape=(..., dimX)
            xbar: 预测状态轨迹 shape=(..., N+1, dimX)
            ubar: 预测控制序列 shape=(..., N, dimU)
            xbar_refs: 参考状态 shape=(..., N+1, dimX)|(..., dimX)
            ubar_refs: 参考控制 None or shape=(..., N, dimU)|(..., dimU)
        Returns:
            total_cost: (...,1) 总代价
        """
        # 预测状态轨迹
        batch_size = x0.shape[0]
        total_cost = torch.zeros(batch_size, device=self.device)

        ls = self.cost_function.stage_cost(
            xbar[..., :-1, :], ubar, xbar_refs[..., :-1, :], ubar_refs
        )  # l(x,u), (...,N,1)
        vfs = self.cost_function.terminal_cost(
            xbar[..., -1:, :], xbar_refs[..., -1:, :]
        )  # (...,1,1)
        cs = torch.cat([ls, vfs], dim=-2)  # (...,N+1,1)
        gammas = shape_rjust(self._gammas, cs)[..., : cs.shape[-2], :]  # (...,N+1,1)
        total_cost = (gammas * cs).sum(dim=-2)  # (...,1)

        # # 阶段代价
        # for t in range(self.horizon):
        #     x1 = xbar[:, t, :]  # [batch_size, dimX]
        #     u1 = ubar[:, t, :]  # [batch_size, dimU]

        #     # 获取参考状态
        #     if xbar_refs.dim() == 2:  # [batch_size, dimX] - 固定参考
        #         state_ref = xbar_refs
        #     else:  # [batch_size, horizon+1, dimX] - 时变参考
        #         state_ref = xbar_refs[:, t, :]

        #     # 获取参考控制
        #     control_ref = None
        #     if ubar_refs is not None:
        #         if ubar_refs.dim() == 2:  # [batch_size, dimU] - 固定参考
        #             control_ref = ubar_refs
        #         else:  # [batch_size, horizon, dimU] - 时变参考
        #             control_ref = ubar_refs[:, t, :]

        #     stage_cost = self.cost_function.stage_cost(x1, u1, state_ref, control_ref)
        #     total_cost += stage_cost

        # # 终端代价
        # terminal_state = xbar[:, -1, :]  # [batch_size, dimX]
        # if xbar_refs.dim() == 2:
        #     terminal_ref = xbar_refs
        # else:
        #     terminal_ref = xbar_refs[:, -1, :]

        # terminal_cost = self.cost_function.terminal_cost(terminal_state, terminal_ref)
        # total_cost += terminal_cost

        return total_cost

    @torch.no_grad()
    def _project_ubar(self):
        """投影控制序列到可行域"""
        if self.control_bounds is not None:
            lb, ub = self.control_bounds
            self._Ubar.data.clip_(lb, ub)

    @torch.no_grad()
    def _set_value(self, name: str, value: torch.Tensor | np.ndarray | Any):
        dst: torch.Tensor = getattr(self, name)
        src = shape_rjust(torch.asarray(value, dtype=dst.dtype, device=dst.device), dst)
        dst.data.copy_(src.data)

    def _set_x0(self, x0: torch.Tensor):
        return self._set_value("_X0", x0)

    def _set_xbar_d(self, xbar_d: torch.Tensor):
        return self._set_value("_Xbar_d", xbar_d)

    def _set_ubar(self, ubar: torch.Tensor):
        return self._set_value("_Ubar", ubar)

    def _set_xbar(self, xbar: torch.Tensor):
        return self._set_value("_Xbar", xbar)

    def solve(
        self,
        x0: torch.Tensor,
        xbar_refs: torch.Tensor,
        ubar_refs: Optional[torch.Tensor] = None,
        xbar_init: Optional[torch.Tensor] = None,
        ubar_init: Optional[torch.Tensor] = None,
        max_iterations: int = 2,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        向量化求解 NMPC 问题
        Args:
            x0: 初始状态 shape=(B, dimX)
            xbar_refs: 期望状态序列 shape=(B,N+1,dimX)
            ubar_refs: 期望控制序列 None or shape=(B,N,dimU)
            xbar_init: [热启动]状态序列初始解, shape=(B,N+1,dimX)
            ubar_init: [热启动]控制序列初始解, shape=(B,N,dimU)
            max_iterations: 最大迭代次数
        Returns:
            optimal_control: (B, N, dimU) 最优控制序列
            info: 优化信息
            - iterations (int): 迭代次数
            - cost_history (list[float]): 代价历史
            - converged (bool): 是否收敛
        """
        self._set_x0(x0)
        self._set_xbar_d(xbar_refs)
        x0 = self._X0
        xbar_refs = self._Xbar_d

        if ubar_refs is not None:
            ubar_refs = ubar_refs.to(self.device)

        # 热启动
        if ubar_init is not None:
            self._set_ubar(ubar_init)
        if xbar_init is not None:
            self._set_xbar(xbar_init)

        # 优化信息
        info = {"iterations": 0, "cost_history": [], "converged": False}

        single_shooting = self.single_shooting
        _reuse_optm = self._reuse_optm
        optr = self._optm

        # 定义闭包函数
        def closure():
            optr.zero_grad()
            # forward
            self._project_ubar()  # 在backward前project
            ubar = self._Ubar

            if single_shooting:
                xbar = self._predict_trajectory(x0, ubar)
                self._set_xbar(xbar.data)
            else:
                xbar = self._Xbar

            # 计算总代价（所有批次的平均）
            costs = self._compute_cost(x0, xbar, ubar, xbar_refs, ubar_refs)
            total_cost = costs.sum()  # separatable opt
            # print(total_cost.item())

            total_cost.backward()

            # 投影梯度
            # with torch.no_grad():
            #     if ubar.grad is not None:
            #         ugrad = ubar.grad
            #         eps = 1e-8
            #         # 处理边界梯度
            #         at_lower = (
            #             ubar.data - self.control_bounds[0] <= eps
            #         )
            #         at_upper = (
            #             ubar.data - self.control_bounds[1] >= -eps
            #         )

            #         grad_mask = torch.ones_like(ugrad)
            #         grad_mask[at_lower & (ugrad > 0)] = 0
            #         grad_mask[at_upper & (ugrad < 0)] = 0

            #         ubar.grad *= grad_mask
            #         pass
            return total_cost

        # 迭代优化
        cost_history: list[float] = info["cost_history"]
        dL_tol = self.optimizer_params["tolerance_change"]
        for iteration in range(max_iterations):
            if _reuse_optm:
                optr = self._optm = self._make_solver()

            try:
                total_cost: torch.Tensor = optr.step(closure)  # type: ignore

                # 记录每个批次的代价
                cost_history.append(total_cost.item())
                info["iterations"] = iteration + 1

                # 检查收敛
                if iteration > 0:
                    _c1 = cost_history[-2]
                    _c2 = cost_history[-1]
                    cost_change = abs(_c1 - _c2)
                    if cost_change < dL_tol:
                        info["converged"] = True
                        break

            except Exception as e:
                warnings.warn(
                    f"Optimization failed at iteration {iteration}: {str(e)}\n{traceback.format_exc()}"
                )
                break

        # print(cost_history[-1], info["iterations"])
        self._project_ubar()
        return self._Ubar.detach().clone(), info

    def get_first_controls(self) -> torch.Tensor:
        """获取第一个控制输入 [batch_size, dimU]"""
        u0 = self._Ubar[..., 0, :]
        return u0.detach().clone()

    def shift(self, steps: int = 1, fix=True):
        """[热启动] 控制(&状态)序列移位"""
        assert steps > 0, ("steps should be positive", steps)
        with torch.no_grad():
            if fix:
                fd = self.dynamics_model
            ubar = self._Ubar
            ubar.data[..., :-steps, :] = ubar.data[..., steps:, :].clone()
            # TODO: 如何更新最后n步控制的初始解?
            # default: U[...,-2,:]
            # NN
            # LQR

            if not self.single_shooting:
                xbar = self._Xbar
                xbar.data[..., :-steps, :] = xbar.data[..., steps:, :].clone()
                if fix:
                    xbar.data[..., -steps:, :] = fd(
                        xbar[..., -(steps + 1) :, :], ubar[..., -steps:, :]
                    )

    def _solver_parameters(self):
        """获取在线优化器参数"""
        ps = [self._Ubar]
        if not self.single_shooting:
            ps.append(self._Xbar)
        for p in ps:
            yield p


class MPCPolicy:
    def __init__(self, solver: BatchMPCSolver):
        """
        Args:
            solver: 向量化 MPC 求解器
        """
        self.solver = solver
        self._device = solver.device
        self._dtype = solver.dtype

    def forward(
        self,
        x0: torch.Tensor,
        xbar_refs: torch.Tensor,
        ubar_refs: Optional[torch.Tensor] = None,
        xbar_init: Optional[torch.Tensor] = None,
        ubar_init: Optional[torch.Tensor] = None,
        max_iterations: int = 2,
    ) -> torch.Tensor:
        """
        执行 MPC 策略
        Args:
            见 BatchMPCSolver.solve()
        Returns:
            optimal_control: (B, N, dimU) 最优控制序列
            info: 优化信息
        """
        rst = self.solver.solve(
            x0, xbar_refs, ubar_refs, xbar_init, ubar_init, max_iterations
        )
        ubar = rst[0]
        return ubar

    def get_first_controls(self) -> torch.Tensor:
        """获取第一个控制输入 shape=(B, dimU)"""
        return self.solver.get_first_controls()
