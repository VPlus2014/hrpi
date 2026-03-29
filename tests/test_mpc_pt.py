# single shooting MPC
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

from dataclasses import dataclass
import traceback
from matplotlib import patches as patches, axes
import torch
import numpy as np
from typing import Sequence

from util_tools import shape_rjust, fit_lim1d


@dataclass
class DrawTurtleBot:
    ax: axes.Axes
    body: patches.Patch | None = None
    arrow: patches.Patch | None = None
    radius: float = 1.0
    color: str = "blue"

    def _patches(self):
        return [self.body, self.arrow]

    def draw(self, x: float, y: float, theta: float):
        for p in self._patches():
            if p:
                p.remove()
        r = self.radius
        color = self.color
        self.body = patches.Circle((x, y), r, color=color, fill=False)
        self.arrow = patches.Arrow(
            x,
            y,
            r * np.cos(theta),
            r * np.sin(theta),
            width=0.2 * r,
            color=color,
        )
        for p in self._patches():
            if p:
                self.ax.add_patch(p)


from codes.agents.policy.mpc_pt.single_shooting import BatchMPCSolver
from codes.agents.policy.mpc_pt.dynamics.car1 import BatchCarModel
from codes.agents.policy.mpc_pt.cost.qp import (
    BatchQPCostFn as VectorizedQPCostFunction,
)
from codes.agents.policy.mpc_pt.dynamics.utils import ode_euler, ode_rk23, ode_rk45
from util_tools import init_seed


def demo():
    init_seed(1008611)
    # 设置设备
    use_cuda = False
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    dtype = torch.float32
    batch_size = 2048  # 并行优化数
    use_line_search = bool(1)  # 是否使用线搜索

    # 系统参数
    simdt = 0.100
    model_pred = BatchCarModel(dt=simdt, solver=ode_rk23)
    model_sim = BatchCarModel(dt=simdt, solver=ode_rk45)
    dimX = model_pred.dimX
    dimU = model_pred.dimU

    # 策略参数
    horizon = 20
    replan_interv = 10  # 重规划间隔
    gamma = 0.99  # 奖励折扣因子
    I_COS = 3
    I_SIN = 4

    max_steps = 1000

    # 代价函数参数
    Q = (
        torch.diag(
            torch.asarray([1.0, 1.0, 0.1, 0.01, 0.01], device=device, dtype=dtype)
        )
        * simdt
    )
    R = torch.eye(dimU, device=device, dtype=dtype) * simdt * 0.01
    Qf = min((1 / max((1 - gamma), 1e-2)), horizon) * Q

    cost_function = VectorizedQPCostFunction(Q, R, Qf)

    # 控制约束
    control_lower = -torch.ones(dimU, device=device, dtype=dtype)
    control_upper = torch.ones(dimU, device=device, dtype=dtype)
    control_bounds = (control_lower, control_upper)

    # 创建向量化 NMPC 优化器
    mpc = BatchMPCSolver(
        dynamics_model=model_pred,
        cost_function=cost_function,
        horizon=horizon,
        control_bounds=control_bounds,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        gamma=gamma,
        **{
            "lr": 1.0,
            "max_iter": 40,
            "tolerance_grad": 1e-6,
            "use_line_search": use_line_search,
        },
    )

    # 批量初始状态和参考轨迹
    def _X_maker(shape: Sequence[int] = (), Vmax=10.0):
        shape = tuple(shape)
        theta0 = torch.rand(shape + (1,)) * (2 * np.pi)
        rmax = Vmax * min(max_steps, 100) * simdt * 0.1
        x0 = torch.rand(shape + (1,)) * rmax
        y0 = torch.rand(shape + (1,)) * rmax
        V0 = torch.rand(shape + (1,)) * Vmax
        return torch.cat([x0, y0, V0, torch.cos(theta0), torch.sin(theta0)], dim=-1)

    initial_states = _X_maker((batch_size,)).to(device, dtype)
    state_refs = shape_rjust(
        _X_maker(Vmax=0).to(device, dtype), mpc._Xbar
    ).broadcast_to(
        mpc._Xbar.shape
    )  # 固定参考

    # 性能测试
    import time
    from codes.utils.time_ext import Timer_Context
    import matplotlib.pyplot as plt

    tmr_infer = Timer_Context("infer")
    tmr_sim = Timer_Context("sim")

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

    bot_r = 0.10
    bot_cur = DrawTurtleBot(ax=ax1, color="cyan", radius=bot_r)
    bot_targ = DrawTurtleBot(ax=ax1, color="red", radius=bot_r)
    bot_pred = DrawTurtleBot(ax=ax1, color="gray", radius=bot_r)

    ax2 = fig.add_subplot(1, 2, 2)
    line_cost = ax2.plot([], [], "-", color="blue", label="cost")[0]
    ax2.set_xlabel("sim step")
    ax2.set_ylabel("cost")
    ax2.legend()

    plt.ion()

    x1 = initial_states
    Xreal = [x1.data.cpu().numpy()[0]]
    costs = []
    for ksim in range(max_steps):
        k_ = ksim % replan_interv
        with tmr_infer:
            if k_ == 0:
                print("-" * 20)
                if ksim > 0:
                    mpc.shift(replan_interv)
                ubar_opt, info = mpc.solve(x1, state_refs, max_iterations=2)

        Xpred = mpc._Xbar.data[0, k_:].cpu().numpy()
        with plt.ioff():

            _vis_Xreal = np.stack(Xreal[-horizon:], axis=0)
            xy_real = _vis_Xreal[:, :2]
            theta_real = np.arctan2(_vis_Xreal[-1, I_SIN], _vis_Xreal[-1, I_COS])
            line_pos_real.set_data(xy_real[:, 0], xy_real[:, 1])
            line_pos_cur.set_data(xy_real[-1:, 0], xy_real[-1:, 1])
            bot_cur.draw(xy_real[-1, 0], xy_real[-1, 1], theta_real)

            xy_pred = Xpred[:, :2]
            line_pos_pred.set_data(xy_pred[:, 0], xy_pred[:, 1])
            theta_pred = np.arctan2(Xpred[-1, I_SIN], Xpred[-1, I_COS])
            bot_pred.draw(xy_pred[-1, 0], xy_pred[-1, 1], theta_pred)

            Xtarg = mpc._Xbar_d.data[0, [k_], :].cpu().numpy()
            xy_targ = Xtarg[:2]
            line_pos_targ.set_data(xy_targ[:, 0], xy_targ[:, 1])
            theta_targ = np.arctan2(Xtarg[-1, I_SIN], Xtarg[-1, I_COS])
            bot_targ.draw(xy_targ[-1, 0], xy_targ[-1, 1], theta_targ)

            ax1.set_xlim(
                *fit_lim1d([xy_real[:, 0], xy_pred[:, 0], xy_targ[:, 0]], rofst=bot_r)
            )
            ax1.set_ylim(
                *fit_lim1d([xy_real[:, 1], xy_pred[:, 1], xy_targ[:, 1]], rofst=bot_r)
            )
            ax1.set_title(f"step {ksim+1}")

            costs.append(info["cost_history"][-1])
            _vis_ks = np.arange(len(costs))[-horizon:]
            _vis_costs = costs[-horizon:]
            line_cost.set_data(_vis_ks, _vis_costs)
            ax2.set_xlim(*fit_lim1d(_vis_ks))
            ax2.set_ylim(*fit_lim1d(_vis_costs))

            fig.canvas.draw()

        plt.pause(0.100)

        with tmr_sim:
            u1 = ubar_opt[..., k_, :]
            print(f"step {ksim+1}: {u1.data[0].cpu().numpy()}")
            pass
            x2: torch.Tensor = model_sim(x1, u1)
            x1 = x2
        Xreal.append(x1.data.cpu().numpy()[0])

        print(f"iterations: {info['iterations']}")
        print(f"converged: {info['converged']}")
        for tmr in [tmr_infer, tmr_sim]:
            print(
                tmr.name,
                f"ms/(batch*step): {tmr.dt*1e3:.1f}/{tmr.t / (ksim+1)*1e3:.1f}",
                f"ms/(unit*step): {tmr.dt*1e3/batch_size:.1f}/{tmr.t / (ksim+1)*1e3/batch_size:.1f}",
                sep=" \t",
            )


if __name__ == "__main__":
    demo()
