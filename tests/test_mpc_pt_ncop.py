# Multiple Shooting MPC as Constrained Nonlinear Optimization Problem
from __future__ import annotations


def _setup():  # 确保项目根节点在 sys.path 中
    import sys
    from pathlib import Path

    __FILE = Path(__file__)
    ROOT = __FILE.parents[1]  # /../..
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    return ROOT


_ROOT = _setup()
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Callable, List, Literal, Optional, Sequence, Tuple, Dict, cast
import numpy as np
from abc import ABC, abstractmethod
from util_tools import shape_rjust

SupportedPenaltyMethod = Literal["augmented_lagrangian", "penalty"]

_AutoGradFuncType = Callable[[Sequence[torch.Tensor], Any], torch.Tensor]


def _default_aux_fn(x: Sequence[torch.Tensor]) -> None:
    pass


def _fmt_iv(iv):
    i, v = iv
    return "{}: {}".format(i, v[0].data.cpu().numpy())


class NonlinearConstrainedOpti:
    """
    基于PyTorch的确定性约束优化问题求解器(Lagrange乘子法)

    支持：
    - 等式约束组 (h(x) = 0)
    - 不等式约束组 (g(x) <= 0)
    - 批量优化
    - 多种优化算法
    """

    DEBUG: bool = True

    def __init__(
        self,
        variables: Sequence[torch.Tensor],
        objective_fn: _AutoGradFuncType,
        equality_fn: Sequence[_AutoGradFuncType] = (),
        inequality_fn: Sequence[_AutoGradFuncType] = (),
        aux_fn: Callable[[Sequence[torch.Tensor]], Any] = _default_aux_fn,
        bounds: Sequence[tuple[torch.Tensor | None, torch.Tensor | None]] | None = None,
        penalty_method: SupportedPenaltyMethod | str = "augmented_lagrangian",
        optimizer_type: str = "lbfgs",
        device: str = "cpu",
        max_iter: int = 40,
        lr=1.0,
        use_slack=False,
        use_line_search=False,
    ):
        r"""
        初始化约束优化器

        Args:
            objective_fn: 目标函数 f(x) -> scalar (for each batch)
            equality_fn: 等式约束函数列表 [h_i(x)]_{i\geq 1}, 对应 $h_i(x)=0,\forall i$
            inequality_fn: 不等式约束函数列表 [g_i(x)]_{i\geq 1}, 对应 $g_i(x)\leq 0,\forall i$
            bounds: 变量边界 (lower_bounds, upper_bounds)
            penalty_method: 惩罚方法 ("augmented_lagrangian", "penalty", "barrier")
            device: 计算设备
            max_iter (int): 优化器迭代次数上限
        """

        self.objective_fn = objective_fn
        self._gs = list(inequality_fn or [])
        self._hs = list(equality_fn or [])
        self._aux_fn = aux_fn
        self.bounds = bounds
        self.penalty_method = penalty_method
        assert penalty_method in SupportedPenaltyMethod.__args__, (
            f"Unsupported penalty method: {penalty_method}",
            "expected one of",
            SupportedPenaltyMethod.__args__,
        )
        self.device = torch.device(device)
        self.dtype = variables[0].dtype
        assert all((v.dtype == self.dtype) for v in variables), (
            "All variables must have the same dtype as",
            (self.dtype),
            "got",
            [v.dtype for v in variables],
        )
        self._var_x = variables

        # 优化参数
        self.rho = 0.1
        self.tolerance = 1e-6
        self.max_iter = max_iter
        self._lam_bound = 1e2
        self._rho_bound = 1.0

        # 初始化拉格朗日乘子
        with torch.no_grad():
            aux = self._aux_fn(self._var_x)
            self._lam_h = self._make_lams(self._hs, aux, 0.0)
            self._lam_g = self._make_lams(self._gs, aux, 1.0)
            if use_slack:
                self._s_g = self._make_slack(self._lam_g)
        self.use_slack = use_slack
        assert not use_slack, NotImplementedError
        self.use_line_search = use_line_search
        self.lr = lr
        self.optimizer_type = optimizer_type

        self.optimizer = self._make_optimizer()
        self._rect_in_closure = False
        self._rt_make_solver = True
        self._rt_make_lams = True

    def _make_optimizer(self) -> optim.Optimizer:
        variables = self._var_x
        optimizer_type = self.optimizer_type
        lr = self.lr
        # 选择优化器
        if optimizer_type == "adam":
            optimizer = optim.Adam(variables, lr=lr)
        elif optimizer_type == "sgd":
            optimizer = optim.SGD(variables, lr=lr)
        elif optimizer_type == "lbfgs":
            optimizer = optim.LBFGS(
                variables,
                lr=lr,
                max_iter=self.max_iter,
                tolerance_grad=self.tolerance,
                tolerance_change=self.tolerance,
                history_size=self.max_iter,
                line_search_fn="strong_wolfe" if self.use_line_search else None,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        return optimizer

    @torch.no_grad()
    def _make_lams(
        self, gs: List[_AutoGradFuncType], aux: Any, default: float = 0.0
    ) -> list[torch.Tensor]:
        lams = []
        _vs = self._var_x
        for i, g in enumerate(gs):
            gx = g(_vs, aux)
            lami = torch.full_like(gx, default)
            lams.append(lami)
        return lams

    def _make_slack(self, lam_g: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        ps = []
        for i, g in enumerate(lam_g):
            p = torch.zeros_like(g)
            ps.append(p)
        return ps

    @torch.no_grad()
    def _apply_bounds(self, x: Sequence[torch.Tensor]):
        """应用边界约束"""
        if self.bounds is not None:
            for _x, (_lb, _ub) in zip(x, self.bounds, strict=True):
                if not (_lb is None and _ub is None):
                    _x.data.clip_(_lb, _ub)
        return x

    def _lagrangian(
        self,
        x: Sequence[torch.Tensor],
        lam_g: Sequence[torch.Tensor],
        lam_h: Sequence[torch.Tensor],
        aux: Any,
    ) -> torch.Tensor:
        r"""L(x,\lambda,\rho)"""
        rho = self.rho

        # 目标函数
        obj_loss = self.objective_fn(x, aux).sum()

        # 等式约束惩罚
        eq_penalty = 0.0
        use_ALM = self.penalty_method == "augmented_lagrangian"
        for lam_i, h_i in zip(lam_h, self._hs, strict=True):
            hx = h_i(x, aux)
            penalty_term = (lam_i * hx).sum()
            if use_ALM:
                penalty_term = penalty_term + (0.5 * rho) * (hx.square().sum())

            eq_penalty = eq_penalty + penalty_term

        # 不等式约束惩罚
        ineq_penalty = 0.0
        for lam_i, g_i in zip(lam_g, self._gs, strict=True):
            gx = g_i(x, aux)
            gx = torch.relu(gx)
            penalty_term = (lam_i * gx).sum()
            ineq_penalty = ineq_penalty + penalty_term

        return obj_loss + eq_penalty + ineq_penalty

    def _check_convergence(
        self,
        x: Sequence[torch.Tensor],
        prev_x: Sequence[torch.Tensor],
        loss: torch.Tensor,
        prev_loss: torch.Tensor,
    ) -> bool:
        """检查收敛性"""
        x_change = max(
            [
                torch.abs(_x1 - _x0).max().item()
                for _x1, _x0 in zip(x, prev_x, strict=True)
            ]
        )
        loss_change = torch.abs(loss - prev_loss).item()

        converged = (x_change < self.tolerance) & (loss_change < self.tolerance)
        return converged

    def _evaluate_constraints(self, x: Sequence[torch.Tensor], aux: Any):
        """评估约束违反程度"""
        eq_violation = [torch.abs(h(x, aux)) for h in self._hs]
        leq_violation = [torch.relu(g(x, aux)) for g in self._gs]
        if len(eq_violation):
            eq_violation = max([_el.max().item() for _el in eq_violation])
        else:
            eq_violation = 0.0
        if len(leq_violation):
            leq_violation = max([_el.max().item() for _el in leq_violation])
        else:
            leq_violation = 0.0
        return eq_violation, leq_violation

    @torch.no_grad()
    def _set_x0(self, x0: Sequence[torch.Tensor]):
        for dst, src in zip(self._var_x, x0, strict=True):
            dst.data.copy_(src.data.to(self.device, self.dtype))

    @torch.no_grad()
    def _x_clone(self) -> Sequence[torch.Tensor]:
        """deepcopy"""
        return [v.detach().clone() for v in self._var_x]

    def solve(
        self,
        x0: Sequence[torch.Tensor],
        max_outer_iterations: int = 10,
        verbose: bool = True,
    ) -> Dict:
        """
        求解

        Args:
            x0: 初始解 shape 与 variables 相容
            lr: 学习率
            optimizer_type: 优化器类型 ("adam", "sgd", "lbfgs")
            max_outer_iterations: 最大外层迭代次数
            verbose: 是否打印详细信息

        Returns:
            优化结果字典
        """
        DEBUG = self.DEBUG
        self._set_x0(x0)
        var_x = self._var_x
        if DEBUG:
            print("x0:", "\n".join(map(_fmt_iv, enumerate(var_x))), sep="\n")
            xbar0 = x0[0]
        _rho_bound = self._rho_bound
        _lam_bound = self._lam_bound
        max_inner_iterations: int = 1

        # 记录优化过程
        history = {
            "objective": [],
            "eq_violation": [],
            "ineq_violation": [],
            "penalty_param": [],
        }

        _rect_in_closure = self._rect_in_closure
        _rt_make_solver = self._rt_make_solver
        _rt_make_lams = self._rt_make_lams
        optimizer = self.optimizer
        if _rt_make_lams:
            with torch.no_grad():
                aux = self._aux_fn(var_x)
            self._lam_g = self._make_lams(self._gs, aux, 1.0)
            self._lam_h = self._make_lams(self._hs, aux, 0.0)
            if self.use_slack:
                self._s_g = self._make_slack(self._lam_g)
            self.rho = 0.1

        converged = False
        # use_ALM = self.penalty_method == "augmented_lagrangian"
        for outer_iter in range(max_outer_iterations):
            prev_x = self._x_clone()
            rho = self.rho

            if _rt_make_solver:
                self.optimizer = self._make_optimizer()
                optimizer = self.optimizer

            # 内层优化
            for inner_iter in range(max_inner_iterations):

                def closure():
                    var_x = self._var_x
                    lambda_h = self._lam_h
                    lambda_g = self._lam_g
                    if _rect_in_closure:
                        self._apply_bounds(var_x)

                    optimizer.zero_grad()
                    aux = self._aux_fn(var_x)
                    loss = self._lagrangian(var_x, lambda_g, lambda_h, aux)
                    total_loss = loss.sum()
                    total_loss.backward()
                    return total_loss

                loss: torch.Tensor = optimizer.step(closure)  # type: ignore
                self._apply_bounds(self._var_x)  # fix variables
                assert all(v.isfinite().all() for v in var_x), "sol with NaN/Inf"
                # var_x = self._apply_bounds(var_x)  # fix variables
                if DEBUG:
                    with torch.no_grad():
                        var_x = self._var_x
                        aux = self._aux_fn(var_x)
                        obj_val = self.objective_fn(var_x, aux)
                        eq_viol, leq_viol = self._evaluate_constraints(var_x, aux)
                        gxs = [g(var_x, aux) for g in self._gs]
                        hxs = [h(var_x, aux) for h in self._hs]

                        xbar, ubar = var_x
                        assert xbar.isfinite().all()
                        assert ubar.isfinite().all()
                        assert loss.isfinite().all()
                        print("xbar0:", xbar0[0].detach().cpu().numpy(), sep="\n")
                        print("xbar:", xbar[0].detach().cpu().numpy(), sep="\n")
                        print("ubar:", ubar[0].detach().cpu().numpy(), sep="\n")
                        # print("ubar", ubar.detach().cpu().numpy())
                        if len(self._gs):
                            print(
                                "gxs:",
                                "\n".join(map(_fmt_iv, enumerate(gxs))),
                                sep="\n",
                            )
                            print(
                                "lam_g:",
                                "\n".join(map(_fmt_iv, enumerate(self._lam_g))),
                                sep="\n",
                            )
                        if len(self._hs):
                            print(
                                "hxs:",
                                "\n".join(map(_fmt_iv, enumerate(hxs))),
                                sep="\n",
                            )
                            print(
                                "lam_h:",
                                "\n".join(map(_fmt_iv, enumerate(self._lam_h))),
                                sep="\n",
                            )
                        print("loss", loss.item())
                        print("obj_val", obj_val.item())
                        print("eq_viol", eq_viol)
                        print("leq_viol", leq_viol)
                        print("rho", rho)
                        print()

            # 评估当前解
            with torch.no_grad():
                aux = self._aux_fn(var_x)
                obj_val = self.objective_fn(var_x, aux)
                eq_viol, leq_viol = self._evaluate_constraints(var_x, aux)

                # 记录历史
                history["objective"].append(obj_val)
                history["eq_violation"].append(eq_viol)
                history["ineq_violation"].append(leq_viol)
                history["penalty_param"].append(rho)

                # 检查收敛
                if outer_iter > 0:
                    prev_obj = history["objective"][-2]
                    new_converged = self._check_convergence(
                        var_x, prev_x, obj_val, prev_obj
                    )
                    converged = converged | new_converged

                # 更新拉格朗日乘数 (增广拉格朗日方法)
                for i, (lam_i, h_i) in enumerate(
                    zip(self._lam_h, self._hs, strict=True)
                ):
                    hx = h_i(var_x, aux)
                    lam_i.data += rho * hx
                    lam_i.data.clip_(-_lam_bound, _lam_bound)
                    if DEBUG:
                        assert lam_i.isfinite().all()
                for i, (lam_i, g_i) in enumerate(
                    zip(self._lam_g, self._gs, strict=True)
                ):
                    gx = g_i(var_x, aux)
                    gx = torch.relu(gx)
                    lam_i.data += rho * gx
                    lam_i.data.clip_(0.0, _lam_bound)  # >=0
                    if DEBUG:
                        assert lam_i.isfinite().all()

                # 更新惩罚参数
                max_violation = max(eq_viol, leq_viol)
                if max_violation > self.tolerance:
                    self.rho = min(self.rho * 1.5, _rho_bound)

                if verbose:
                    print(
                        f"Iteration {outer_iter}: "
                        f"Obj={obj_val.item():.6f}, "
                        f"Eq_viol={eq_viol:.6f}, "
                        f"Ineq_viol={leq_viol:.6f}",
                        f"Penalty={self.rho:.6f}",
                    )

                # 如果所有批次都收敛，提前停止
                if converged:
                    break
        sol = {
            "x": var_x,
            "objective": obj_val.item(),
            "eq_violation": float(eq_viol),
            "ineq_violation": float(leq_viol),
            "converged": converged,
            "iterations": outer_iter + 1,
            "history": history,
            "lam_g": self._lam_g,
            "lam_h": self._lam_h,
            "rho": self.rho,
        }
        return sol


def bquad(x: torch.Tensor, Q: torch.Tensor):
    r"""
    二次型函数 $x\cdot Q x$
    Args:
        x: 输入变量 (..., dimX)
        Q: 二次型矩阵 (..., dimX, dimX)
    Returns:
        二次型函数值 (..., 1)
    """
    y = (x * (x @ Q)).sum(dim=-1, keepdim=True)
    return y


# 使用示例
def demo():
    """使用示例"""
    from tests.test_mpc_pt import BatchCarModel, ode_rk23, ode_euler, ode_rk45
    from util_tools import init_seed
    from codes.utils.time_ext import Timer_Context
    import matplotlib.pyplot as plt

    init_seed(1008611)
    use_cuda = False
    device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
    th_float = torch.float64
    simdt = 0.100
    model_pred = BatchCarModel(dt=simdt, solver=ode_rk23)
    model_sim = BatchCarModel(dt=simdt, solver=ode_rk45)
    dimX = model_pred.dimX
    dimU = model_pred.dimU
    horizon = 20
    batch_size = 1
    I_COS = 3
    I_SIN = 4
    _Q = torch.eye(dimX, device=device, dtype=th_float) * simdt
    _R = torch.eye(dimU, device=device, dtype=th_float) * simdt * 0.01
    xlb = torch.zeros(dimX, device=device, dtype=th_float) - 1000
    xub = torch.zeros(dimX, device=device, dtype=th_float) + 1000
    ulb = torch.zeros(dimU, device=device, dtype=th_float) - 1
    uub = torch.zeros(dimU, device=device, dtype=th_float) + 1

    def _X_maker(shape: Sequence[int] = (), Vmax=10.0):
        shape = tuple(shape)
        theta0 = torch.rand(shape + (1,)) * (2 * np.pi)
        rmax = 100.0
        x0 = torch.rand(shape + (1,)) * rmax
        y0 = torch.rand(shape + (1,)) * rmax
        V0 = torch.rand(shape + (1,)) * Vmax
        return torch.cat([x0, y0, V0, torch.cos(theta0), torch.sin(theta0)], dim=-1)

    var_Xbar = nn.Parameter(
        torch.zeros((batch_size, horizon + 1, dimX), device=device, dtype=th_float)
    )
    var_Ubar = nn.Parameter(
        torch.zeros((batch_size, horizon, dimU), device=device, dtype=th_float)
    )
    param_X0 = torch.zeros((batch_size, dimX), device=device, dtype=th_float)
    param_Xbar_d = torch.zeros((batch_size, horizon + 1, dimX), device=device, dtype=th_float)

    xlb = shape_rjust(xlb, var_Xbar)
    xub = shape_rjust(xub, var_Xbar)
    ulb = shape_rjust(ulb, var_Ubar)
    uub = shape_rjust(uub, var_Ubar)
    _Q = shape_rjust(_Q, var_Xbar)
    _R = shape_rjust(_R, var_Ubar)

    use_predX = False

    def objective(vs: Sequence[torch.Tensor], aux: Any):
        xbar, ubar = vs
        # x2p: torch.Tensor = aux[0]
        # xbar = torch.cat([xbar[..., 0:1, :], x2p], dim=-2)
        if use_predX:
            xbar = aux[1]
        xerr = xbar - param_Xbar_d
        assert xerr.shape == xbar.shape
        cx = bquad(xerr, shape_rjust(_Q, xerr)).sum()
        cu = bquad(ubar, shape_rjust(_R, ubar)).sum()
        return cx + cu

    def init_constraint(vs: Sequence[torch.Tensor], aux: Any):
        xbar, ubar = vs
        h_x0 = xbar[..., 0, :] - param_X0  # (...,dimX)
        return h_x0  # .abs()

    def transition_constraint(vs: Sequence[torch.Tensor], aux: Any):
        xbar, ubar = vs

        x2 = xbar[..., 1:, :]
        x2t: torch.Tensor = aux[0]
        h_x12 = x2 - x2t
        return h_x12  # .abs()

    def get_aux(vs: Sequence[torch.Tensor]):
        xbar, ubar = vs
        x1 = xbar[..., :-1, :]
        x2p: torch.Tensor = model_pred(x1, ubar)
        aux = [x2p]
        if use_predX:
            xbarp = torch.cat([xbar[..., 0:1, :], x2p], dim=-2)
            aux.append(xbarp)
        return aux

    def g_xlb(vs: Sequence[torch.Tensor], aux: Any):
        xbar, ubar = vs
        g = xlb - xbar
        assert g.shape == xbar.shape
        return g

    def g_xub(vs: Sequence[torch.Tensor], aux: Any):
        xbar, ubar = vs
        g = xbar - xub
        assert g.shape == xbar.shape
        return g

    def g_ulb(vs: Sequence[torch.Tensor], aux: Any):
        xbar, ubar = vs
        g = ulb - ubar
        assert g.shape == ubar.shape
        return g

    def g_uub(vs: Sequence[torch.Tensor], aux: Any):
        xbar, ubar = vs
        g = ubar - uub
        assert g.shape == ubar.shape
        return g

    def _single_shooting(xbar: torch.Tensor, ubar: torch.Tensor):
        n = ubar.shape[-2]
        for k in range(n):
            xbar.data[..., k + 1, :] = model_pred(xbar[..., k, :], ubar[..., k, :])
        return xbar

    # 批量初始解
    param_Xbar_d.data.copy_(shape_rjust(_X_maker(), param_Xbar_d))
    param_X0.data.copy_(shape_rjust(_X_maker((batch_size,)), param_X0))
    print("param_Xbar_d:", param_Xbar_d[0].cpu().numpy(), sep="\n")
    print("param_X0:", param_X0[0].cpu().numpy(), sep="\n")
    sol_x_0 = [
        torch.rand_like(var_Xbar, requires_grad=False),
        ulb + torch.rand_like(var_Ubar, requires_grad=False) * (uub - ulb),
    ]  # 随机初始解
    sol_x_0[0] = _single_shooting(sol_x_0[0], sol_x_0[1])

    # 创建优化器
    optimizer = NonlinearConstrainedOpti(
        variables=[var_Xbar, var_Ubar],
        objective_fn=objective,
        equality_fn=[
            init_constraint,
            transition_constraint,
        ],
        inequality_fn=[
            # g_xlb,
            # g_xub,
            # g_ulb,
            # g_uub,
        ],
        aux_fn=get_aux,
        device=device,
        bounds=[(xlb, xub), (ulb, uub)],
        max_iter=40,
        use_line_search=True,
    )

    from util_tools import fit_lim1d

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    line_pos_cur = ax1.plot([], [], "o", color="cyan", label="cur")[0]
    line_pos_real = ax1.plot([], [], "-", color="blue", label="real")[0]
    line_pos_pred = ax1.plot([], [], "--", color="gray", label="pred")[0]
    line_pos_targ = ax1.plot([], [], "*", color="red", label="targ")[0]

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    ax1.set_aspect("equal")

    ax2 = fig.add_subplot(1, 2, 2)
    line_cost = ax2.plot([], [], "-", color="blue", label="cost")[0]
    ax2.set_xlabel("sim step")
    ax2.set_ylabel("cost")
    ax2.legend()

    bot_r = 1.0

    plt.ion()

    tmr_sol = Timer_Context("solve")
    tmr_sim = Timer_Context("sim")
    max_steps = 1000
    optimizer.DEBUG = bool(0)
    X1 = sol_x_0[0][..., 0, :].to(device, th_float)  # (B,)
    assert X1.shape == param_X0.shape
    Xreal = [X1.data[0].cpu().numpy()]
    prev_sol_x = sol_x_0
    costs = []

    def _shift(x: torch.Tensor, n: int = 1):
        assert n > 0
        x.data[..., :-n, :] = x.data[..., n:, :].clone()

    for ksim in range(max_steps):
        k_ = 0
        with tmr_sol:
            # 优化
            if ksim > 0:
                _shift(prev_sol_x[0])
                _shift(prev_sol_x[1])

            param_X0.data.copy_(X1.data)
            result = optimizer.solve(
                x0=prev_sol_x, max_outer_iterations=20, verbose=False
            )

        sol_J: float = result["objective"]
        prev_sol_x: List[torch.Tensor] = result["x"]
        sol_xbar, sol_ubar = prev_sol_x
        Xpred = sol_xbar.data[0, k_:].cpu().numpy()

        print("\n=== 优化结果 ===")
        print(f"最优解: {result['x']}")
        print(f"目标函数值: {result['objective']}")
        print(f"等式约束违反: {result['eq_violation']}")
        print(f"不等式约束违反: {result['ineq_violation']}")
        print(f"收敛状态: {result['converged']}")
        print(f"迭代次数: {result['iterations']}")
        print(f"耗时: {tmr_sol.dt*1e3:.0f} ms/batch {tmr_sol.dt*1e3/batch_size:.0f} ms/unit")

        with plt.ioff():
            _vis_Xreal = np.stack(Xreal[-horizon:], axis=0)
            xy_real = _vis_Xreal[:, :2]
            # theta_real = np.arctan2(_vis_Xreal[-1, I_SIN], _vis_Xreal[-1, I_COS])
            line_pos_real.set_data(xy_real[:, 0], xy_real[:, 1])
            line_pos_cur.set_data(xy_real[-1:, 0], xy_real[-1:, 1])
            # bot_cur.draw(xy_real[-1, 0], xy_real[-1, 1], theta_real)

            xy_pred = Xpred[:, :2]
            line_pos_pred.set_data(xy_pred[:, 0], xy_pred[:, 1])
            # theta_pred = np.arctan2(Xpred[-1, I_SIN], Xpred[-1, I_COS])
            # bot_pred.draw(xy_pred[-1, 0], xy_pred[-1, 1], theta_pred)

            Xtarg = param_Xbar_d.data[0, [k_], :].cpu().numpy()
            xy_targ = Xtarg[:2]
            line_pos_targ.set_data(xy_targ[:, 0], xy_targ[:, 1])
            # theta_targ = np.arctan2(Xtarg[-1, I_SIN], Xtarg[-1, I_COS])
            # bot_targ.draw(xy_targ[-1, 0], xy_targ[-1, 1], theta_targ)

            ax1.set_xlim(
                *fit_lim1d([xy_real[:, 0], xy_pred[:, 0], xy_targ[:, 0]], rofst=bot_r)
            )
            ax1.set_ylim(
                *fit_lim1d([xy_real[:, 1], xy_pred[:, 1], xy_targ[:, 1]], rofst=bot_r)
            )
            ax1.set_title(f"step {ksim+1}")

            costs.append(sol_J)
            _vis_ks = np.arange(len(costs))[-horizon:]
            _vis_costs = costs[-horizon:]
            line_cost.set_data(_vis_ks, _vis_costs)
            ax2.set_xlim(*fit_lim1d(_vis_ks))
            ax2.set_ylim(*fit_lim1d(_vis_costs))

            fig.canvas.draw()

        plt.pause(0.100)

        with tmr_sim:
            u1 = sol_ubar[..., k_, :]
            print(f"step {ksim+1}: {u1.data[0].cpu().numpy()}")
            pass
            X2: torch.Tensor = model_sim(X1, u1)
            X1 = X2
        Xreal.append(X1.data[0].cpu().numpy())
    return result


if __name__ == "__main__":
    demo()
