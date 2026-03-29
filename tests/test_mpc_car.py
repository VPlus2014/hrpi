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

from collections import OrderedDict
from copy import deepcopy
from heapq import nsmallest
import time
from timeit import timeit
from typing import (
    Any,
    Iterable,
    Sequence,
    SupportsIndex,
    TypeVar,
    Union,
    cast,
    overload,
)
import gymnasium as gym
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from numba import njit, jit

# from torch.utils.tensorboard.writer import SummaryWriter
from numpy.typing import NDArray

import util_tools
import casadi as ca


CaMatLike = Union[ca.DM, ca.MX, ca.SX]


def as_numpy(v: CaMatLike | NDArray) -> "NDArray[np.floating]":
    if isinstance(v, np.ndarray):
        return v
    if isinstance(v, (ca.SX, ca.MX)):
        v = ca.DM(v)  # convert SX to DM iff all elements are set

    if isinstance(v, ca.DM):
        v = v.full()
    return np.asarray(v)


class SimpleCarModel:

    def __init__(self, batch_size: int = 1):
        self.dimX = 4
        self.dimU = 1
        self.V = 10.0
        self.batch_size = batch_size

    def dynamics(self, X: Sequence, U: Sequence):
        # x = X[0]
        # y = X[1]
        cost = X[2]
        sint = X[3]
        u = U[0]
        V = self.V
        dot_x = V * cost
        dot_y = V * sint
        dcos = -sint * u
        dsin = cost * u
        dotX = (dot_x, dot_y, dcos, dsin)
        return dotX

    def fd_maker(self):
        from ca_mpc_controllers.ca_math import ode_rk23

        bsz = self.batch_size
        dt: ca.MX = ca.MX.sym("dt")  # type: ignore
        X: ca.MX = ca.MX.sym("X", self.dimX, bsz)  # type: ignore
        U: ca.MX = ca.MX.sym("U", self.dimU, bsz)  # type: ignore
        X2 = ode_rk23(self.dynamics, ca.vertsplit(X), ca.vertsplit(U), dt)
        X2 = ca.vcat(X2)
        assert X2.shape == X.shape, X2.shape
        fd = ca.Function("fd", [X, U, dt], [X2])
        return fd


def draw_traj(named_xbars: dict[str, Any], dimX: int, horizon: int, fig=None):
    # from mpl_toolkits.mplot3d.art3d import Line3D
    # from mpl_toolkits.mplot3d import Axes3D

    if fig is None:
        fig = plt.figure()
    # ax = cast(Axes3D, fig.add_subplot(1, 1, 1, projection="3d"))
    ax = fig.add_subplot(1, 1, 1)
    ax.cla()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # ax.set_zlabel("z")
    ax.set_aspect("equal")
    # ax.invert_zaxis()  # Z 轴反向
    clr = ["r", "g", "b", "y", "m", "c"]
    for im, (name, Xbar) in enumerate(named_xbars.items()):
        Xbar = as_numpy(Xbar).reshape(dimX, horizon + 1, -1)
        pos = Xbar[:2, :, 0]
        ci = clr[im % len(clr)]
        ax.plot(pos[0, :], pos[1, :], color=ci, alpha=0.5, label=f"traj_{name}")
        ax.scatter(pos[0, 0], pos[1, 0], color=ci, label=f"start_{name}")
    ax.legend()
    return fig, ax


def main():
    from ca_mpc_controllers.ca_math import traj_single_shooting

    # seed = int(time.time())
    seed = 1008611
    rng = np.random.default_rng(seed)
    horizon = 50
    bsz = 2

    dt = 0.100
    model = SimpleCarModel(batch_size=bsz)
    dimX = model.dimX
    dimU = model.dimU
    fd_ = model.fd_maker()
    fd = lambda x, u: fd_(x, u, dt)
    single_shooting = False
    max_iter = 20
    continue_if_max_iter = True  # 非常关键, warmstart

    opti = ca.Opti()

    var_Xbar = opti.variable(dimX, (horizon + 1) * bsz)
    var_Ubar = opti.variable(dimU, (horizon * bsz))
    param_Xbar_d = opti.parameter(dimX, (horizon + 1) * bsz)
    param_X0 = opti.parameter(dimX, bsz)

    gs = []
    gs.append(param_X0 == var_Xbar[:, :bsz])
    for k in range(horizon):
        i1 = slice(k * bsz, (k + 1) * bsz)
        i2 = slice(i1.start + bsz, i1.stop + bsz)
        x1 = var_Xbar[:, i1]
        u1 = var_Ubar[:, i1]
        x2t = fd(x1, u1)
        gs.append(var_Xbar[:, i2] == x2t)
    _0 = np.zeros((dimU, horizon * bsz), dtype=np.float64)
    ulb = -1.0 + _0
    uub = 1.0 + _0
    gs.append(opti.bounded(ulb, var_Ubar, uub))

    gamma = 1 - 1 / 100

    Qk = np.eye(dimX) * 1.0  # type: ignore
    Rk = np.eye(dimU) * dt  # type: ignore
    cost = 0
    for k in range(horizon):
        i1 = slice(k * bsz, (k + 1) * bsz)
        uk = var_Ubar[:, i1]
        ek = var_Xbar[:, i1] - param_Xbar_d[:, i1]  # (dimX, B)
        cx = ca.sum(ek * (Qk @ ek))
        cu = ca.sum(uk * (Rk @ uk))
        gammak = gamma**k
        cost += (cx + cu) * gammak

    i1 = slice(-bsz, None)
    ek = var_Xbar[:, i1] - param_Xbar_d[:, i1]
    cx = ca.sum(ek * (Qk @ ek)) * (1 / (1 - gamma))
    cost += cx

    opti.minimize(cost)
    for g in gs:
        opti.subject_to(g)

    solver_opts = {
        "ipopt.max_iter": max_iter,
        "ipopt.print_level": 0,
        "ipopt.warm_start_init_point": "yes",  # 热启动
        "ipopt.acceptable_tol": 1e-8,
        "ipopt.acceptable_obj_change_tol": 1e-6,
        "ipopt.hessian_approximation": "limited-memory",  # 使用拟牛顿法（有限内存近似）
        "print_time": 1,
    }

    opti.solver("ipopt", solver_opts)

    x0 = rng.random((dimX, bsz))
    opti.set_value(param_X0, x0)
    #
    du = (ulb + (uub - ulb) * rng.random((dimU, horizon * bsz))) * dt
    ini_ubar = np.cumsum(du.reshape(dimU, horizon, bsz), axis=1).reshape(dimU, -1)
    ini_ubar = np.clip(ini_ubar, ulb, uub)

    xbar_d = np.zeros((dimX, (horizon + 1) * bsz))
    xbar_d = traj_single_shooting(fd, x0, ini_ubar, horizon, bsz, out_xbar=xbar_d)
    opti.set_value(param_Xbar_d, xbar_d)

    ini_ubar = ini_ubar * rng.random((dimU, horizon * bsz))
    # ini_ubar = np.clip(ini_ubar, ulb, uub)
    opti.set_initial(var_Ubar, ini_ubar)
    #
    ini_xbar = np.zeros((dimX, (horizon + 1) * bsz))
    ini_xbar[:, :bsz] = x0
    for k in range(horizon):
        i1 = slice(k * bsz, (k + 1) * bsz)
        i2 = slice(i1.start + bsz, i1.stop + bsz)
        x1 = ini_xbar[:, i1]
        x2 = fd(x1, ini_ubar[:, i1])
        ini_xbar[:, i2] = x2
    if single_shooting:
        pass
    else:
        opti.set_initial(var_Xbar, ini_xbar)

    fig = None
    plt.ion()
    for itr in range(100):
        try:
            sol: ca.OptiSol = opti.solve()
            loss = sol.value(cost)
            print("suc", loss)
            ubar = sol.value(var_Ubar)
            xbar = sol.value(var_Xbar)
        except Exception as e:
            opti_: ca.Opti = opti.debug
            loss = opti_.value(cost)
            print("max_iter reached", loss)
            ubar: np.ndarray = opti_.value(var_Ubar)
            xbar: np.ndarray = opti_.value(var_Xbar)

        ubar = ubar.reshape(dimU, horizon * bsz)
        # xbar_ss = np.zeros_like(xbar)
        # xbar_ss = traj_single_shooting(fd, x0, ubar, horizon, bsz, out_xbar=xbar_ss)

        fig = draw_traj(
            OrderedDict(
                [
                    ("required", xbar_d),
                    ("guess", ini_xbar),
                    ("sol_MS", xbar),
                    # ("sol_SS", xbar_ss),
                ]
            ),
            dimX,
            horizon,
            fig=fig,
        )[0]
        plt.pause(1.0)

        if continue_if_max_iter:
            opti.set_initial(var_Ubar, ubar)
            opti.set_initial(var_Xbar, xbar)


if __name__ == "__main__":
    main()
