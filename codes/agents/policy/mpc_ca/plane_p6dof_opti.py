from __future__ import annotations
import contextlib
from pathlib import Path

from matplotlib import pyplot as plt

if __name__ == "__main__":
    raise RuntimeError("don't directly run", Path(__file__))


class Timer_Context(contextlib.ContextDecorator):
    """上下文计时器"""

    # Usage: @TimerL() decorator or 'with TimerL():' context manager
    def __init__(self, name: str, t=0.0):
        """上下文计时器"""
        self.name = name
        """name"""
        self.t = t
        """total time, unit: sec"""
        self.dt = 0.0
        """last delta-time, unit: sec"""
        self.__nstack = 0

    def reset(self):
        self.t = 0.0
        self.dt = 0.0
        self.__nstack = 0

    def __enter__(self):
        self.push()
        return self

    def __exit__(self, type, value, traceback):
        self.pop()

    def time(self):
        return time.time()

    def add_dt(self, dt: float):
        self.dt += dt  # delta-time
        self.t += dt  # accumulate dt

    def push(self):
        if self.__nstack == 0:
            self.start = self.time()
            self.dt = 0.0
        self.__nstack += 1

    def pop(self):
        assert self.__nstack > 0, "Timer stack underflow"
        self.__nstack -= 1
        if self.__nstack == 0:
            dt = self.time() - self.start  # delta-time
            self.add_dt(dt)


import math
import time
from typing import (
    Any,
    Callable,
    List,
    OrderedDict,
    Sequence,
    TypeVar,
    cast,
    no_type_check,
)
import casadi as ca
from casadi import SX, MX, DM
from matplotlib.pylab import norm
import numpy as np
from numpy.typing import NDArray
from .ca_math import *
from .ca_nn import CANNModule
from . import ca_nn

_T = TypeVar("_T")


def calc_range1d(pts: Sequence[_T] | np.ndarray):
    vs = np.hstack([np.ravel(p) for p in pts])  # type: ignore
    return np.min(vs), np.max(vs)


def fit_lim(range: Sequence[_T] | np.ndarray, rspan=0.05):
    a, b = calc_range1d(range)
    c = (b + a) * 0.5
    r = (b - a) * 0.5 * (1 + rspan)
    if r == 0:
        r = 1e-3
    return c - r, c + r


def draw_traj(named_xbars: dict[str, Any], dimX: int, horizon: int):
    from mpl_toolkits.mplot3d.art3d import Line3D
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = cast(Axes3D, fig.add_subplot(1, 1, 1, projection="3d"))
    ax.set_aspect("equal")
    ax.invert_zaxis()  # Z 轴反向
    clr = ["r", "g", "b", "y", "m", "c"]
    for im, (name, Xbar) in enumerate(named_xbars.items()):
        Xbar = as_numpy(Xbar).reshape(dimX, horizon + 1, -1)
        pos = Xbar[:3, :, 0]
        ci = clr[im % len(clr)]
        ax.plot(
            pos[0, :], pos[1, :], pos[2, :], color=ci, alpha=0.5, label=f"traj_{name}"
        )
        ax.scatter(pos[0, 0], pos[1, 0], pos[2, 0], color=ci, label=f"start_{name}")
    ax.legend()
    return fig, ax


class _InfSeq(Sequence):
    """infinite sequence of 1"""

    def __getitem__(self, item: int):
        return 1

    def __len__(self):
        return math.inf

    def __iter__(self):
        return self

    def __next__(self):
        # warning: no stop condition
        return 1


def _vsplit_keepdim(v) -> Sequence:
    if isinstance(v, CaMatLike):
        y = ca.vertsplit(v)
    elif isinstance(v, Sequence):
        return v
    else:
        if not isinstance(v, np.ndarray):
            v = np.asarray(v)
        y = np.vsplit(v, len(v))
        # assert all(z.shape[0] == 1 for z in y)
    return y


class PlaneP6DOFModel:
    def __init__(
        self,
        batch_size=1,
        g: float = 9.81,
        use_gravity=True,
        use_MX=False,
        ode_solver: ODESolverType[T_ArithElem_co] = ode_rk23,
        rectify_state=True,
    ):
        self._g = float(g)
        self._use_gravity = use_gravity
        self.batch_size = batch_size
        self.dimX = len(self.X_split(_InfSeq()))
        self.dimU = len(self.U_split(_InfSeq()))
        self.use_MX = use_MX
        self.ca_f = self.f_maker(use_MX)
        r"""(X_t,U_t) -> \dot X_{t}"""
        self.ode_solver = ode_solver
        self.rectify_state = rectify_state
        self.ca_fd = self.fd_maker(
            use_MX, ode_solver=ode_solver, rectify_state=rectify_state
        )
        r"""(X_t,U_t,dt) -> X_{t+1}"""

    @classmethod
    def X_split(cls, X: Sequence[_T] | _T) -> Tuple[_T, ...]:
        X = _vsplit_keepdim(X)  # type: ignore
        assert isinstance(X, Sequence), TypeError(f"X is not a sequence: {X}", type(X))
        x, y, z = X[0], X[1], X[2]
        V = X[3]
        q0, q1, q2, q3 = X[4], X[5], X[6], X[7]
        return x, y, z, V, q0, q1, q2, q3

    @classmethod
    def X_merge(cls, x, y, z, V, q0, q1, q2, q3):
        return x, y, z, V, q0, q1, q2, q3

    @classmethod
    def U_split(cls, U: Sequence[_T] | _T) -> Tuple[_T, ...]:
        U = _vsplit_keepdim(U)  # type: ignore
        assert isinstance(U, Sequence), TypeError(f"U is not a sequence: {U}", type(U))
        nx, ny, nz, omega1 = U[0], U[1], U[2], U[3]
        return nx, ny, nz, omega1

    @classmethod
    def U_merge(cls, nx, ny, nz, omega1):
        return nx, ny, nz, omega1

    def dynamics(
        self,
        X: Sequence[T_ArithElem_co],
        U: Sequence[T_ArithElem_co],
    ) -> tuple[T_ArithElem_co, ...]:
        g = self._g
        x, y, z, V, q0, q1, q2, q3 = self.X_split(X)
        nx, ny, nz, p = self.U_split(U)
        Qeb = [q0, q1, q2, q3]
        Qbe = quat_conj(Qeb)
        if self._use_gravity:
            g_b = quat_rot(Qbe, [0, 0, g])
            ax_b = nx * g + g_b[0]
            ay_b = ny * g + g_b[1]
            az_b = nz * g + g_b[2]
        Vinv = 1.0 / V
        P_ = p
        Q_ = -Vinv * az_b
        R_ = Vinv * ay_b
        dotq0, dotq1, dotq2, dotq3 = quat_mul(Qeb, [0, 0.5 * P_, 0.5 * Q_, 0.5 * R_])
        dotV = ax_b
        dotx, doty, dotz = quat_rot(Qeb, [V, 0, 0])
        dotX = self.X_merge(dotx, doty, dotz, dotV, dotq0, dotq1, dotq2, dotq3)
        return dotX  # type: ignore

    def f_maker(self, use_MX=False, batch_size: int | None = None):
        _sym = MX.sym if use_MX else SX.sym
        n = batch_size or self.batch_size
        # t = _sym("t", 1, n)  # type: ignore
        X = _sym("X", self.dimX, n)  # type: ignore
        U = _sym("U", self.dimU, n)  # type: ignore
        dotX = self.dynamics(X, U)
        dotX = ca.vcat(dotX)
        assert dotX.shape == X.shape, f"state dim mismatch: {dotX.shape}!={X.shape}"
        f = ca.Function("f", [X, U], [dotX])
        return f

    def fd_maker(
        self,
        use_MX=False,
        batch_size: int | None = None,
        ode_solver: ODESolverType | None = None,
        rectify_state: bool | None = None,
    ):
        _sym = MX.sym if use_MX else SX.sym
        n = batch_size or self.batch_size
        X = _sym("X", self.dimX, n)  # type: ignore
        U = _sym("U", self.dimU, n)  # type: ignore
        dt = _sym("dt", 1, 1)  # type: ignore
        if ode_solver is None:
            ode_solver = self.ode_solver

        X2 = ode_solver(self.dynamics, ca.vertsplit(X), ca.vertsplit(U), dt)
        rectify_state = rectify_state or self.rectify_state
        if rectify_state:
            X2 = self.X_rectify(X2)
        X2 = ca.vertcat(*X2)
        assert X2.shape == X.shape, f"state dim mismatch: {X2.shape}!={X.shape}"
        fd = ca.Function("fd", [X, U, dt], [X2])
        return fd

    # @classmethod
    def X_rectify(self, X):
        x, y, z, V, q0, q1, q2, q3 = self.X_split(X)
        # V = max(V, 1e-3)
        q0, q1, q2, q3 = quat_normalize([q0, q1, q2, q3], 1e-3)
        X_ = self.X_merge(x, y, z, V, q0, q1, q2, q3)
        return X_


class MPController:

    DEBUG = False

    Vf: Callable[[CaMatLike | np.ndarray, CaMatLike | np.ndarray], CaMatLike]
    r"""
    state value without time weight $\gamma^t$
    Args:
        X: shape=(dimX, B)
        X_d: shape=(dimX, B)
    Returns:
        vs: $\bar{V}(X_d-X)$ shape=(1, B)
    """

    def __init__(
        self,
        model: PlaneP6DOFModel,
        dt: float,
        horizon: int = 2,
        max_iter=10,
        print_time=True,
        single_shooting=False,
        nn_vf: CANNModule | None = None,  # (dimX,B)->(1,B)
        nn_u_guess: CANNModule | None = None,  # (dimX,B)->(dimU,B)
        gamma: float = 0.95,
        use_quansi_newton=False,
        debug=DEBUG,
    ):
        self.DEBUG = debug
        assert not single_shooting, "single_shooting is deprecated"

        self.model = model
        self.horizon = horizon
        self.gamma = float(gamma)
        assert 0 <= gamma <= 1, f"expect gamma in [0,1], got {gamma}"
        self.dt = dt
        self.single_shooting = single_shooting
        self.use_nn_Vf = nn_vf is not None
        if self.use_nn_Vf:
            # from ca_nn import MLP

            # self._Vf_kern = MLP(dimX, 1, [128, 128], fixed_batch_size=bsz)
            self._Vf_kern = nn_vf  # (dimX,B)->(1,B)
        else:
            self._Vf_kern = None
        self.use_u_guess = nn_u_guess is not None
        if self.use_u_guess:
            self._mu_kern = nn_u_guess
        else:
            self._mu_kern = None
        # assert not nn_u_guess, NotImplementedError("'use_nn_guess' not supported yet")

        dimX = model.dimX
        dimU = model.dimU
        bsz = model.batch_size
        # assert bsz == 1, NotImplementedError("batch_size > 1 not supported yet")

        self.opti = opti = ca.Opti()
        self.var_Ubar: MX = opti.variable(dimU, horizon * bsz)
        self.var_Xbar: MX = opti.variable(dimX, (horizon + 1) * bsz)
        self.param_X0: MX = opti.parameter(dimX, bsz)  # [参数]初始状态
        self.param_Xbar_d: MX = opti.parameter(
            dimX, (horizon + 1) * bsz
        )  # [参数]目标状态
        # self.param_ts = opti.parameter(1, horizon + 1)  # [参数]时间序列

        self._make_solver(
            max_iter=max_iter,
            print_time=print_time,
            use_quansi_newton=use_quansi_newton,
        )
        return

    def _get_meta_constraints(self):
        """simple constraints for one-step&unit state and control"""
        rmax = 10000.0
        Vmin = 10.0
        Vmax = 500.0
        model = self.model
        lb_xk = np.asarray(
            model.X_merge(-rmax, -rmax, -rmax, Vmin, -1, -1, -1, -1)
        ).reshape(-1, 1)
        ub_xk = np.asarray(model.X_merge(rmax, rmax, rmax, Vmax, 1, 1, 1, 1)).reshape(
            -1, 1
        )
        assert np.all(lb_xk <= ub_xk), ("expect lb_xk <= ub_xk, got", lb_xk, ub_xk)
        omega1_max = math.pi / 4
        omega1_min = -omega1_max
        nx_max = 5
        nx_min = -1
        ny_max = 0.2
        ny_min = -ny_max
        nz_max = 1
        nz_min = -8
        lb_uk = np.asarray(model.U_merge(nx_min, ny_min, nz_min, omega1_min)).reshape(
            -1, 1
        )
        ub_uk = np.asarray(model.U_merge(nx_max, ny_max, nz_max, omega1_max)).reshape(
            -1, 1
        )
        assert np.all(lb_uk <= ub_uk), ("expect lb_uk <= ub_uk, got", lb_uk, ub_uk)
        return lb_xk, ub_xk, lb_uk, ub_uk

    def _make_constraints(self, var_Ubar, var_Xbar, param_X0, param_Xd):
        gs = []
        single_shooting = self.single_shooting
        model = self.model
        dt = self.dt
        dimX = model.dimX
        dimU = model.dimU
        bsz = model.batch_size
        lb_xk, ub_xk, lb_uk, ub_uk = self._get_meta_constraints()
        lb_xk_ = lb_xk.reshape(dimX, 1).repeat(bsz, axis=1)  # (dimX,B)
        ub_xk_ = ub_xk.reshape(dimX, 1).repeat(bsz, axis=1)
        lb_uk_ = lb_uk.reshape(dimU, 1).repeat(bsz, axis=1)  # (dimU,B)
        ub_uk_ = ub_uk.reshape(dimU, 1).repeat(bsz, axis=1)

        # lb_xk = ca.MX(lb_xk)
        # ub_xk = ca.MX(ub_xk)
        # lb_uk = ca.MX(lb_uk)
        # ub_uk = ca.MX(ub_uk)

        opti = self.opti
        bsz = model.batch_size
        gs = []

        # 初始状态约束
        gs.append((var_Xbar[:, :bsz] == param_X0))

        for k in range(self.horizon):
            i1 = slice(k * bsz, (k + 1) * bsz)
            i2 = slice((k + 1) * bsz, (k + 2) * bsz)
            xk = var_Xbar[:, i1]
            uk = var_Ubar[:, i1]

            # 状态转移约束
            x2targ = model.ca_fd(xk, uk, dt)
            x2pred = var_Xbar[:, i2]
            gs.append((x2pred == x2targ))

            # 后继状态约束
            gx2 = opti.bounded(lb_xk_, x2pred, ub_xk_)
            gs.append(gx2)

            # 控制约束
            gu1 = opti.bounded(lb_uk_, uk, ub_uk_)
            gs.append(gu1)

        # lbg = ca.vertcat(*lbg)
        # ubg = ca.vertcat(*ubg)
        return gs, lb_xk, ub_xk, lb_uk, ub_uk

    def _vf_maker(self, use_MX=True):
        _sym = MX.sym if use_MX else SX.sym
        dimX = self.model.dimX
        bsz = self.model.batch_size
        sym_xf = _sym("xf", dimX, bsz)  # type: ignore
        sym_xfd = _sym("xfd", dimX, bsz)  # type: ignore
        ek = sym_xf - sym_xfd
        if self.use_nn_Vf:
            assert self._Vf_kern is not None, "use_nn_Vf=True but no Vf_kern provided"
            vfs = self._Vf_kern(ek)
        else:
            vfs = ek * (self.Qw @ ek)  # one-step cost
            vfs = ca.sum1(vfs)
            gamma = self.gamma
            rr = (1 / (1 - gamma)) if gamma < 1 else self.horizon
            vfs = vfs * rr
        Vf = ca.Function("Vf", [sym_xf, sym_xfd], [vfs])  # -> shape=(1,B)
        return Vf

    def _make_cost(self, Ubar, Xbar, Xbar_d, uk_bound) -> MX:
        # fd = self.model.cs_fd
        model = self.model
        dimU = model.dimU
        dimX = model.dimX
        horizon = self.horizon
        bsz = model.batch_size
        gamma = self.gamma
        assert (
            Xbar.shape == Xbar_d.shape
        ), f"Xbar.shape!=Xbar_d.shape: {Xbar.shape}!= {Xbar_d.shape}"

        dx = 1
        dz = 0.8 * dx
        self._Qw = np.asarray([1 / dx**2] * 2 + [1 / dz**2] + [1 / 1**2] + [1e-2] * 4)
        self.Qw = np.diag(self._Qw)  # (dimX,dimX)
        assert self.Qw.shape == (
            dimX,
            dimX,
        ), f"Qw.shape!= ({dimX},{dimX}): {self.Qw.shape}"

        assert uk_bound.shape == (
            dimU,
        ), f"uk_bound.shape!= ({dimU},): {uk_bound.shape}"
        self._Rw = 1 / uk_bound**2
        self.Rw = np.diag(self._Rw)  # (dimU,dimU)
        assert self.Rw.shape == (
            dimU,
            dimU,
        ), f"Rw.shape!= ({dimU},{dimU}): {self.Rw.shape}"

        # 状态预测误差

        # Qk = np.diag(self.Qw)
        # Rk = np.diag(self.Rw)
        cost = 0
        gamma_k = 1.0
        for k in range(horizon):
            i1 = slice(k * bsz, (k + 1) * bsz)
            i2 = slice(i1.stop, i1.stop + bsz)
            ek = Xbar[:, i1] - Xbar_d[:, i2]
            uk = Ubar[:, i1]
            cx = ca.sum(ek * (self.Qw @ ek))
            cu = ca.sum(uk * (self.Rw @ uk))
            cost += (cx + cu) * gamma_k
            gamma_k *= gamma

        # 终端代价
        self.Vf = self._vf_maker()  # type: ignore
        i2 = slice((horizon) * bsz, (horizon + 1) * bsz)
        vfs = self.Vf(Xbar[:, i2], Xbar_d[:, i2])
        vfs = ca.sum(vfs)
        cost += vfs * gamma_k

        # reduce
        cost = cost / (bsz * horizon)
        return cost

    def _make_solver(
        self, max_iter: int, print_time: bool = True, use_quansi_newton=False
    ):
        var_Xbar = self.var_Xbar
        var_Ubar = self.var_Ubar
        param_X0 = self.param_X0
        param_Xd = self.param_Xbar_d
        opti = self.opti
        assert (
            var_Xbar.shape == param_Xd.shape
        ), f"Xbar.shape!=Xbar_d.shape: {var_Xbar.shape}!= {param_Xd.shape}"
        # if param_Xd.shape[1] < var_Xbar.shape[1]:
        #     param_Xd = ca.repmat(param_Xd, 1, self.horizon + 1)

        # solx = self.solx_merge(var_Ubar, var_Xbar)
        gs, lb_xk, ub_xk, lb_uk, ub_uk = self._make_constraints(
            var_Ubar, var_Xbar, param_X0, param_Xd
        )
        uk_bound = np.maximum(np.abs(lb_uk), np.abs(ub_uk)).ravel()
        cost = self._make_cost(var_Ubar, var_Xbar, param_Xd, uk_bound)
        self.var_cost = cost

        opti.minimize(cost)
        for g in gs:
            opti.subject_to(g)

        # 编译求解器
        solver_opts = {
            "ipopt.max_iter": max_iter,
            "ipopt.print_level": 0,
            "ipopt.warm_start_init_point": "yes",  # 热启动
            "ipopt.acceptable_tol": 1e-8,
            "ipopt.acceptable_obj_change_tol": 1e-6,
            "print_time": int(print_time),
        }
        if use_quansi_newton:
            solver_opts["ipopt.hessian_approximation"] = (
                "limited-memory"  # 使用拟牛顿法（有限内存近似）
            )

        opti.solver("ipopt", solver_opts)
        return

    def predict_control(self, e1, u1_ref=None):
        """predict/guess one-step control with NN/padding"""
        assert (
            e1.shape[-1] == self.model.batch_size
        ), f"x.shape[-1]!= {self.model.batch_size}"
        if self.use_u_guess:
            assert (
                self._mu_kern is not None
            ), "use_nn_guess=True but no mu_kern provided"
            u1_ = self._mu_kern(e1)
            # raise NotImplementedError("not supported yet")
            return u1_
        else:
            assert u1_ref is not None, "u1_ref is None"
            return u1_ref

    def load_prev_sol(self, prev_sol: ca.OptiSol):
        if prev_sol is None:
            return
        opti = self.opti
        opti.set_initial(self.var_Ubar, prev_sol.value(self.var_Ubar))
        # if self.single_shooting:
        # opti.set_initial(self.var_Xbar, prev_sol.value(self.var_Xbar))

    def reset(self, X0: np.ndarray | DM, Xbar_d: np.ndarray | DM):
        """generate initial guess before starting optimization"""
        opti = self.opti
        opti.set_value(self.param_X0, X0)
        opti.set_value(self.param_Xbar_d, Xbar_d)

        xbar = DM.zeros(self.var_Xbar.shape)  # type: ignore
        ubar = DM.zeros(self.var_Ubar.shape)  # type: ignore
        self.shift(ubar, xbar, X0, Xbar_d, shift_step=self.horizon)

    def shift(
        self,
        ubar_prev: (
            np.ndarray | DM
        ),  # last predicted U_{k:k+N-1}, may be changed in-place
        xbar_prev: np.ndarray | DM,  # last predicted X_{k:k+N}, may be changed in-place
        x0: np.ndarray | DM,  # real X_{k+s}
        xbar_d: np.ndarray | DM,  # desired X_{k+s:k+s+N}
        shift_step: int = 1,  # s
    ):
        self.opti.set_value(self.param_X0, x0)
        self.opti.set_value(self.param_Xbar_d, xbar_d)
        assert shift_step >= 0, f"shift_len should be non-negative, got {shift_step}"
        bsz = self.model.batch_size
        slen = shift_step * bsz
        ubar_prev = np.roll(as_numpy(ubar_prev), -slen, axis=1)  # [N-s:N] is invalid
        xbar_prev = np.roll(as_numpy(xbar_prev), -slen, axis=1)  # [N-s:N+1] is invalid
        xbar_prev[:, :bsz] = x0
        horizon = self.horizon
        dt = self.dt
        fd = lambda x, u: self.model.ca_fd(x, u, dt)
        for k in range(horizon - shift_step, horizon):
            i1 = slice(k * bsz, (k + 1) * bsz)
            i2 = slice(i1.stop, i1.stop + bsz)
            x1 = xbar_prev[:, i1]
            u1 = ubar_prev[:, i1]
            e1 = x1 - xbar_d[:, i1]
            u1 = self.predict_control(e1, u1)
            x2 = fd(x1, u1)
            ubar_prev[:, i1] = u1
            xbar_prev[:, i2] = x2

        self.opti.set_initial(self.var_Ubar, ubar_prev)
        self.opti.set_initial(self.var_Xbar, xbar_prev)

    def calc_control(
        self,
        x0: DM,  # new initial state (dimX,B)
        xbar_d: DM,  # new desired state sequence (dimX,(N+1)*B)
        echo=False,
    ):
        """
        get control&state sequence with optimization

        Args:
            x0 (DM): initial state (dimX,B)
            xbar_d (DM): desired state sequence (dimX,(N+1)*B)
        Returns:

            u0_opt (np.ndarray): optimized control sequence (dimU,B).

            cost (np.ndarray): optimized cost.

            ubar (np.ndarray): optimized state sequence (dimX,N*B).

            xbar (np.ndarray): optimized state sequence (dimX,(N+1)*B).
        """
        tmr_set = Timer_Context("set")
        tmr_solve = Timer_Context("solve")
        tmr_get = Timer_Context("get")
        opti = self.opti
        with tmr_set:
            opti.set_value(self.param_X0, x0)
            opti.set_value(self.param_Xbar_d, xbar_d)

        # optimization
        try:
            with tmr_solve:
                sol: ca.OptiSol = opti.solve()
            with tmr_get:
                ubar = sol.value(self.var_Ubar)
                xbar = sol.value(self.var_Xbar)
                cost = sol.value(self.var_cost)
        except RuntimeError as e:  # max_iter reached
            with tmr_get:
                opti_: ca.Opti = opti.debug
                ubar = opti_.value(self.var_Ubar)
                xbar = opti_.value(self.var_Xbar)
                cost = opti_.value(self.var_cost)
        except Exception as e:
            print(f"calc_control error: {e}")
            raise e
        with tmr_get:
            ubar = cast(NDArray[np.floating], ubar)
            xbar = cast(NDArray[np.floating], xbar)
            cost = cast(NDArray[np.floating], cost)
            ubar = ubar.reshape(self.var_Ubar.shape)
            xbar = xbar.reshape(self.var_Xbar.shape)

            bsz = self.model.batch_size
            u0_opt = ubar[:, :bsz]

        if echo or self.DEBUG:
            print(
                {tmr.name: f"{tmr.t*1e3:.0f}" for tmr in [tmr_set, tmr_solve, tmr_get]}
            )
        return u0_opt, cost, ubar, xbar

    def solx_split(self, solx: DM):
        if self.single_shooting:
            U = solx
            X = None
        else:
            szU = self.model.dimU * self.horizon
            U = solx[:szU]
            X = solx[szU:]
            X = X.reshape(self.var_Xbar.shape)
        U = U.reshape(self.var_Ubar.shape)
        return U, X


def _test_step(sim_fd, t, x, u, simdt, nitr=1000, n_unit=1, name=""):
    x = _vsplit_keepdim(x)
    u = _vsplit_keepdim(u)
    _wallt0 = time.time()
    for i in range(nitr):
        for j in range(n_unit):
            x1 = sim_fd(t, x, u, simdt)
    dtwall = time.time() - _wallt0
    speed_ratio = simdt / (dtwall / nitr)
    print(f"{dtwall/nitr:.6f}s/step, speed_ratio:{speed_ratio:.3e}, arch:{name}")


def test_ode2():
    """
    速度:
    单例下 MX~SX>>scipy~原始rk4
    多例下 SX<MX<<scipy~原始rk4, 分水岭大约 batch_size=150
    """
    n = 150
    model = PlaneP6DOFModel(batch_size=n)
    simdt = 0.050

    def _f_np(x: Sequence, u: Sequence):
        y = model.dynamics(x, u)  # Sequence[ shape=(1,n) ]
        return y

    def fd_hand_maker(f_kern):

        def _fd_hand(t, x: Sequence, u: Sequence, dt):
            y = ode_rk45(f_kern, x, u, dt)
            return y

        return _fd_hand

    def fd_scipy_maker(f_kern):
        mode = -1
        from scipy.integrate import solve_ivp, odeint, RK45

        def f4scipy(t, x: np.ndarray, u: Sequence[np.ndarray]):
            x = x.reshape(-1, n)
            dx = f_kern(_vsplit_keepdim(x), _vsplit_keepdim(u))
            dx = np.vstack(dx)
            dx = dx.reshape(-1)
            return dx

        def fd(
            t: np.ndarray, x: Sequence[np.ndarray], u: Sequence[np.ndarray], dt: float
        ):
            f = lambda t, x: f4scipy(t, x, u)
            x = np.vstack(x).reshape(-1)  # type: ignore
            if mode == 0:
                sol = solve_ivp(f, [0, dt], x, args=(u,), method="RK45")
                y = sol.y[:, -1]
            elif mode == 1:
                sol = odeint(f, x, [0, dt], args=(u,), tfirst=True, hmax=dt)
                y = sol[-1]
            else:
                sol = RK45(f, 0, x, t_bound=dt)
                y = sol.y
            # y
            return y

        return fd

    x0 = np.random.rand(model.dimX, n)
    u0 = np.random.rand(model.dimU, n)

    fd_sci = fd_scipy_maker(_f_np)
    fd_hand = fd_hand_maker(_f_np)
    fd_ca_MX_kern = model.fd_maker(use_MX=True)
    fd_ca_SX_kern = model.fd_maker(use_MX=False)

    def fd_ca_MX(t, x: Sequence, u: Sequence, dt) -> Sequence:
        x_ = ca.vcat(x)
        u_ = ca.vcat(u)
        y = fd_ca_MX_kern(x_, u_, dt)
        return ca.horzsplit(y)

    def fd_ca_SX(t, x: Sequence, u: Sequence, dt) -> Sequence:
        x_ = ca.vcat(x)
        u_ = ca.vcat(u)
        y = fd_ca_SX_kern(x_, u_, dt)
        return ca.horzsplit(y)

    t0_dm = DM(0.0)
    x0_dm = DM(x0)
    u0_dm = DM(u0)
    nitr = 10
    nunit = 10
    _test_step(
        fd_hand, t0_dm, x0, u0, simdt, nitr=nitr, n_unit=nunit, name="hand-crafted"
    )
    _test_step(
        fd_ca_MX, t0_dm, x0_dm, u0_dm, simdt, nitr=nitr, n_unit=nunit, name="casadi_MX"
    )
    _test_step(
        fd_ca_SX, t0_dm, x0_dm, u0_dm, simdt, nitr=nitr, n_unit=nunit, name="casadi_SX"
    )
    _test_step(fd_sci, t0_dm, x0, u0, simdt, nitr=nitr, n_unit=nunit, name="scipy")


def _test_jit():
    print(ca.CasadiMeta.compiler())
    raise NotImplementedError
    symX = MX.sym("x", 2)
    print(symX.shape)
    # symY = MX.sym("y", 2)
    A = np.asarray(
        [
            [0, -1],
            [1, 0],
        ]
    )
    symY = A @ symX
    f = ca.Function("f", [symX], [symY], {"compiler": "g++"})
    x = np.random.rand(2)
    y = f(x)
    print(y)

    body = "r[0] = x[0];" + "while (r[0]<s[0]) {" + " r[0] *= r[0];" + "}"
    sp = ca.Sparsity.scalar()
    f = ca.Function.jit(
        "f123123", body, ["x", "s"], ["r"], [sp, sp], [sp], {"compiler": "shell"}
    )
    print(f)

    body = "\n".join(
        map(
            lambda x: str(x) + ";",
            [
                # "double a=x[0]+y[0]+z[0]",
                # "a*=a",
                "dotx[0]=x[0]",
                "doty[0]=y[0]",
                "dotz[0]=z[0]",
            ],
        )
    )
    sp = ca.Sparsity.scalar()
    f2 = ca.Function.jit(
        "f2",
        body,
        ["x", "y", "z"],
        ["dotx", "doty", "dotz"],
        [sp] * 3,
        [sp] * 3,
        {"compiler": "shell"},
    )
    x = np.random.rand(3)
    y = f2(x)
    print(y)
    return f


def demo():
    test_ode2()

    simdt = 0.050  # 反馈仿真步长
    ctrl_interval = 10
    horizon = 10  # 预测时域
    ctrldt = simdt * ctrl_interval
    print("horizon: {:.3f}s".format(horizon * ctrldt))
    use_single_shooting = bool(0)
    max_steps = int(240 / ctrldt)
    solver_max_iter = 40  # 单步优化求解次数限制
    solver_quansi_nicolson = bool(0)  # 拟牛顿法, MX模式下建议关闭
    solver_print_time = bool(0)
    use_MX = bool(1)
    viz = bool(1)
    bsz = 1  # 批容量
    use_nn_VF = bool(0)
    use_nn_mu = bool(0)
    gamma = 0.99
    if bsz > 1:
        assert use_MX, "MX is required while batch_size > 1"

    model_pred = PlaneP6DOFModel(
        batch_size=bsz, use_MX=use_MX, ode_solver=ode_euler, rectify_state=False
    )
    model_real = PlaneP6DOFModel(
        batch_size=bsz, use_MX=use_MX, ode_solver=ode_rk45, rectify_state=True
    )
    dimU = model_pred.dimU
    dimX = model_pred.dimX
    if use_nn_VF:
        nn_vf = ca_nn.MLP(dimX, 1, [128, 128], use_MX=use_MX, fixed_batch_size=bsz)
    else:
        nn_vf = None
    if use_nn_mu:
        # linear regulizer
        nn_mu = ca_nn.MLP(dimX, dimU, use_MX=use_MX, fixed_batch_size=bsz)
    else:
        nn_mu = None
    policy = MPController(
        model_pred,
        horizon=horizon,
        dt=ctrldt,
        single_shooting=use_single_shooting,
        nn_vf=nn_vf,
        nn_u_guess=nn_mu,
        max_iter=solver_max_iter,
        print_time=solver_print_time,
        use_quansi_newton=solver_quansi_nicolson,
        gamma=gamma,
        debug=True,
    )

    # 参考轨迹
    #
    pe_ref = np.array([-1000, 500.0, -2000])
    V_ref = 120.0
    qeb_ref = np.array([1.0, 0.0, 0.0, 0.0])
    #
    xd0 = (
        np.hstack([pe_ref, V_ref, qeb_ref]).reshape(-1, 1).repeat(bsz, axis=1)
    )  # (dimX,B)
    assert xd0.shape == (dimX, bsz)
    from .traj import generate_cylindrical_spiral

    pd = generate_cylindrical_spiral(
        1000.0,
        10.0,
        omega=(2 * math.pi) / 1.0,
        t0=0.0,
        tf=horizon * ctrldt,
        n=max_steps + horizon,
    )  # (1,T,3)
    pd = pd.repeat(bsz, axis=0)  # (B,T,3)
    pd = pd.transpose((2, 1, 0))  # (3,T,B)
    pd = pd.reshape(3, -1)  # (3,T*B)
    xd_all = xd0.repeat(max_steps + horizon, axis=1)  # (dimX,T*B)
    xd_all[:3, :] += pd
    xd_all = xd_all.reshape(dimX, -1)
    Xd_np = xd_all.reshape(dimX, -1, bsz)  # (dimX,T,B)

    x0 = (
        np.hstack([[0.0, 0.0, 0.0], 100 + V_ref, [1.0, 0.0, 0.0, 0.0]])
        .reshape(-1, 1)
        .repeat(bsz, axis=1)
    )  # (dimX,B)

    Xsim_np = np.zeros((dimX, (max_steps + 1), bsz))
    Xsim_dm = DM.zeros((dimX, (max_steps + 1) * bsz))  # type: ignore
    Xsim_dm[:, :bsz] = x0  # 初始状态
    Xsim_np[:, 0] = x0
    ts = np.zeros(max_steps + 1)
    costs = np.zeros_like(ts)

    def sim_fd(t, x, u) -> DM:  # 状态转移反馈
        return model_real.ca_fd(x, u, simdt)  # type: ignore

    dts_infer = [0.0]
    dts_step = [0.0]
    tmr_infer = Timer_Context("infer")
    tmr_sim = Timer_Context("sim")
    tmr_render = Timer_Context("render")

    if viz:
        plt.ion()
        np.set_printoptions(precision=4, suppress=True)
        from mpl_toolkits.mplot3d.art3d import Line3D
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax1 = cast(Axes3D, fig.add_subplot(1, 2, 1, projection="3d"))
        ax1.set_aspect("equal")
        ax1.invert_zaxis()  # Z 轴反向
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_aspect("auto")

        line_ref = cast(
            Line3D, ax1.plot([], [], [], "--", color="pink", label="ref")[0]
        )
        # ax1.scatter(pe_ref[0], pe_ref[1], pe_ref[2], c="cyan", label="ref", marker="*")
        line_los = cast(
            Line3D, ax1.plot([], [], [], "--", color="cyan", label="los")[0]
        )
        line_pos = cast(Line3D, ax1.plot([], [], [], label="pos", linestyle="-")[0])
        # if not use_single_shooting:
        line_pos_pred = cast(
            Line3D,
            ax1.plot([], [], [], color="gray", label="pos_pred", linestyle="--")[0],
        )
        # line_pp = ax1.plot([], [], [], label="pos_pred", linestyle="--")[0]
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        lines_eb: List[Line3D] = [
            cast(
                Line3D,
                ax1.plot(
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    label="axis" + ("XYZ"[i]),
                    color="rgb"[i],
                    alpha=0.8,
                )[0],
            )
            for i in range(3)
        ]
        ebs = np.eye(3)
        ax1.legend()

        line2 = ax2.plot([], [], label="cost")[0]
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Cost")
        # plt.show()

        def _rend_pos(line: Line3D, xs, ys, zs):
            line.set_data(xs, ys)
            line.set_3d_properties(zs)

        def _rend_body(X):
            pass

    def _render(k, tk, u_opt, costk, ubar, xbar, Xsim_np, Xd_np, V0p, Vfp):
        if not viz:
            return
        Xsim_np0 = Xsim_np[..., 0]
        Xd_np0 = Xd_np[..., 0]
        # 真实轨迹
        k0 = max(0, k - 2 * horizon)
        k2 = min(k + 1 + horizon, Xd_np.shape[1])

        pos_real = Xsim_np0[:3, k0 : k + 2]
        _rend_pos(line_pos, *pos_real)

        pos_ref = Xd_np0[:3, k0:k2]
        _rend_pos(line_ref, *pos_ref)

        rwin = np.max(np.abs(pos_real[:, 0] - pos_real[:, -1]))
        rwin = max(1, rwin)

        # 预测轨迹
        pos_pred = as_numpy(xbar[:3, :]).reshape(3, horizon + 1, bsz)[:, :, 0]
        _rend_pos(line_pos_pred, *pos_pred)

        # 体轴系
        X2_np = Xsim_np0[:, [k + 1]]
        _rst = model_pred.X_split(X2_np)
        _rst = [_.item() for _ in _rst]
        x, y, z, V, q0, q1, q2, q3 = _rst
        Qeb = [q0, q1, q2, q3]
        axscale = rwin * 0.2
        for iax in range(3):
            ebi = quat_rot(Qeb, ebs[iax])
            _rend_pos(
                lines_eb[iax],
                [x, x + ebi[0] * axscale],
                [y, y + ebi[1] * axscale],
                [z, z + ebi[2] * axscale],
            )

        losR = norm([x, y, z] - pe_ref)
        _rend_pos(line_los, [x, pe_ref[0]], [y, pe_ref[1]], [z, pe_ref[2]])

        tag_global = False
        if tag_global:
            idxs = slice(0, k + 2)
            xs = Xsim_np0[0, idxs]
            ys = Xsim_np0[1, idxs]
            zs = Xsim_np0[2, idxs]
            xmin = min(np.min(xs), pe_ref[0])
            xmax = max(np.max(xs), pe_ref[0])
            ymin = min(np.min(ys), pe_ref[1])
            ymax = max(np.max(ys), pe_ref[1])
            zmin = min(np.min(zs), pe_ref[2])
            zmax = max(np.max(zs), pe_ref[2])
        else:
            xmin = x - rwin
            xmax = x + rwin
            ymin = y - rwin
            ymax = y + rwin
            zmin = z - rwin
            zmax = z + rwin
        ax1.set_xlim(*fit_lim([xmin, xmax]))
        ax1.set_ylim(*fit_lim([ymin, ymax]))
        ax1.set_zlim(*fit_lim([zmin, zmax]))

        _rst = model_pred.U_split(u_opt[:, 0:1])
        _rst = [_.item() for _ in _rst]
        nx, ny, nz, p = _rst
        n_b = np.asarray([nx, ny, nz])
        pos = np.asarray([x, y, z])
        ez = pe_ref[-1] - z
        ax1.set_title(
            "\n".join(
                [
                    f"Time: {tk:.02f} s",
                    f"|LOS|:{losR:.02f}, zd-z:{ez:.02f}, V:{V:.03f}",
                    f"pos:{pos}",
                    f"n_b:{n_b}, P_b:{p:.03g}",
                ]
            )
        )
        # ax.relim(True)
        # ax.autoscale(tight=True)

        idxs = slice(k0, k + 1)
        # ts_ = ts[idxs]
        ts_ = np.arange(k0, k + 1)
        costs_ = costs[idxs]
        line2.set_data(ts_, costs_)
        # ax2.set_xticks(ts_)
        ax2.set_xlim(*fit_lim([ts_[0], ts_[-1]]))
        ax2.set_ylim(*fit_lim(costs_))
        ax2.set_title(
            "\n".join(
                [
                    f"Cost: {costk:.4}",
                    f"V0 pred: {V0p:.4}",  # 预测当前代价
                    f"Vf pred: {Vfp:.4}",  # 预测终端代价
                    "dt_infer: {:.3f}ms".format(np.mean(dts_infer[idxs]) * 1e3),
                    "dt_step:  {:.3f}ms".format(np.mean(dts_step[idxs]) * 1e3),
                ]
            )
        )

    def render(k, tk, u_opt, costk, ubar, xbar, Xsim_np, Xd_np, V0p, Vfp):
        if not viz:
            return
        print("render start")
        with tmr_render:
            with plt.ioff():
                _render(k, tk, u_opt, costk, ubar, xbar, Xsim_np, Xd_np, V0p, Vfp)

            # fig.tight_layout()
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.010)

        print(f"render: {tmr_render.dt*1e3:.0f}ms")

    xbk_dm = DM(x0)
    start_time = time.time()
    tk_in = time.time() - start_time

    # warmstart
    xbar_d_dm = DM(xd_all[:, : (horizon + 1) * bsz])
    policy.reset(xbk_dm, xbar_d_dm)
    k = 0
    tk = k * ctrldt
    # itrs_warm = np.clip(horizon // 2, 1, 5)
    itrs_warm = 1
    for k_warm in range(itrs_warm):
        rst = policy.calc_control(xbk_dm, xbar_d_dm)
        u_opt, costk, ubar, xbar = rst
        print(f"warmup {k_warm}:", "cost", costk)
        policy.opti.set_initial(policy.var_Ubar, ubar)
        policy.opti.set_initial(policy.var_Xbar, xbar)
        # draw_traj(
        #     dict([("sol", xbar), ("demand", xbar_d_dm)]),
        #     dimX=dimX,
        #     horizon=horizon,
        # )
        # plt.pause(0.01)
        V0p = as_numpy(policy.Vf(xbk_dm, xbar_d_dm[:, :bsz]))[..., 0].item()
        Vfp = as_numpy(policy.Vf(xbar[:, -bsz:], xbar_d_dm[:, -bsz:]))[..., 0].item()
        if viz:
            render(k, tk, u_opt, costk, ubar, xbar, Xsim_np, Xd_np, V0p, Vfp)

    # 滚动优化
    X2 = x0
    for k in range(max_steps):
        tk = k * ctrldt

        with tmr_infer:
            xbar_d_dm = DM(xd_all[:, k * bsz : (k + horizon + 1) * bsz])
            policy.shift(ubar, xbar, X2, xbar_d_dm)
            rst = policy.calc_control(xbk_dm, xbar_d_dm)
            u_opt, costk, ubar, xbar = rst
            costk = as_numpy(costk).item()

            V0p = as_numpy(policy.Vf(xbk_dm, xbar_d_dm[:, :bsz]))[..., 0].item()
            Vfp = as_numpy(policy.Vf(xbar[:, -bsz:], xbar_d_dm[:, -bsz:]))[
                ..., 0
            ].item()

        dts_infer.append(tmr_infer.dt)

        ts[k] = tk
        costs[k] = costk

        # 状态转移
        with tmr_sim:
            i1 = slice(k * bsz, (k + 1) * bsz)
            i2 = slice(i1.stop, i1.stop + bsz)
            X1 = Xsim_dm[:, i1]
            X2 = X1
            for _ in range(ctrl_interval):
                X2 = sim_fd(tk, X2, u_opt)

            Xsim_dm[:, i2] = X2
            Xsim_np[:, k + 1] = X2
            xbk_dm = X2

        dts_step.append(tmr_sim.dt)

        print(
            " ".join(
                [
                    f"k:{k}",
                    f"tk:{tk:.3f}s",
                    f"dt_infer:{dts_infer[-1]*1e3:.0f}/{np.mean(dts_infer[-horizon:])*1e3:.0f}",
                    f"dt_sim:{dts_step[-1]*1e3:.0f}/{np.mean(dts_step[-horizon:])*1e3:.0f}",
                ]
            )
        )

        if viz:
            render(k, tk, u_opt, costk, ubar, xbar, Xsim_np, Xd_np, V0p, Vfp)

    if viz:
        plt.show(block=True)
