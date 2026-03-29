from __future__ import annotations
from pathlib import Path

if __name__ == "__main__":
    raise RuntimeError("don't directly run", Path(__file__))

# TODO:
# - 向量化(需要解决矩阵结构下时间和分组的联合索引)

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
from ca_math import *

_T = TypeVar("_T")


def as_numpy(v: CaMatLike | NDArray) -> "NDArray[np.floating]":
    if isinstance(v, np.ndarray):
        return v
    if isinstance(v, (SX, MX)):
        v = ca.DM(v)  # convert SX to DM iff all elements are set

    if isinstance(v, DM):
        v = v.full()
    return np.asarray(v)


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


class PesudoDOF6Model:
    def __init__(
        self,
        batch_size=1,
        g: float = 9.81,
        use_gravity=True,
        use_MX=False,
    ):
        self._g = float(g)
        self._use_gravity = use_gravity
        self.batch_size = batch_size
        self.dimX = len(self.X_split(_InfSeq()))
        self.dimU = len(self.U_split(_InfSeq()))
        self.use_MX = use_MX
        self.ca_f = self._f_maker(use_MX)
        self.ca_fd = self._fd_maker(use_MX)
        """(t,X_t,U_t)->X_{t+1}"""

    @classmethod
    def X_split(cls, X: Sequence[_T]) -> Tuple[_T, ...]:
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
    def U_split(cls, U: Sequence[_T]) -> Tuple[_T, ...]:
        U = _vsplit_keepdim(U)  # type: ignore
        assert isinstance(U, Sequence), TypeError(f"U is not a sequence: {U}", type(U))
        nx, ny, nz, omega1 = U[0], U[1], U[2], U[3]
        return nx, ny, nz, omega1

    @classmethod
    def U_merge(cls, nx, ny, nz, omega1):
        return nx, ny, nz, omega1

    def dynamics(
        self,
        t: T_ArithElem_co,
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

    @no_type_check
    def _f_maker(self, use_MX=False):
        _sym = MX.sym if use_MX else SX.sym
        n = self.batch_size
        t = _sym("t", 1, n)
        X = _sym("X", self.dimX, n)
        U = _sym("U", self.dimU, n)
        dotX = self.dynamics(t, X, U)
        dotX = ca.vcat(dotX)
        assert dotX.shape == X.shape, f"state dim mismatch: {dotX.shape}!={X.shape}"
        f = ca.Function("f", [t, X, U], [dotX])
        return f

    @no_type_check
    def _fd_maker(self, use_MX=False):
        _sym = MX.sym if use_MX else SX.sym
        n = self.batch_size
        t = _sym("t", 1, n)
        X = _sym("X", self.dimX, n)
        U = _sym("U", self.dimU, n)
        dt = _sym("dt")
        X2 = ode_rk45(self.dynamics, t, ca.vertsplit(X), ca.vertsplit(U), dt)
        X2 = self.X_rectify(X2)
        X2 = ca.vertcat(*X2)
        assert X2.shape == X.shape, f"state dim mismatch: {X2.shape}!={X.shape}"
        fd = ca.Function("fd", [t, X, U, dt], [X2])
        return fd

    # @classmethod
    def X_rectify(self, X):
        x, y, z, V, q0, q1, q2, q3 = self.X_split(X)
        # V = max(V, 1e-3)
        q0, q1, q2, q3 = quat_normalize([q0, q1, q2, q3], 1e-3)
        X_ = self.X_merge(x, y, z, V, q0, q1, q2, q3)
        return X_


class MPController:

    def __init__(
        self,
        model: PesudoDOF6Model,
        dt: float,
        horizon: int = 2,
        max_iter=10,
        print_time=True,
        use_single_shooting=True,
        use_nn_vf=False,
        use_nn_guess=False,
        use_MX=True,
    ):
        self.model = model
        self.horizon = horizon
        self.dt = dt
        self.use_single_shooting = use_single_shooting
        self.use_nn_Vf = use_nn_vf
        self.use_nn_guess = use_nn_guess
        assert not use_nn_guess, NotImplementedError("'use_nn_guess' not supported yet")

        dimX = model.dimX
        bsz = model.batch_size
        assert bsz == 1, NotImplementedError("batch_size > 1 not supported yet")
        _sym = MX.sym if use_MX else SX.sym
        self.var_Xbar = _sym("X", dimX, (horizon + 1) * bsz)
        self.var_Ubar = _sym("U", model.dimU, horizon * bsz)
        self.param_X0 = _sym("x0", dimX * bsz)  # [参数]初始状态
        self.param_Xd = _sym("xref", dimX, (horizon + 1) * bsz)  # [参数]目标状态
        self.param_ts = _sym("t", 1, (horizon + 1) * bsz)

        if self.use_nn_Vf:
            from ca_nn import MLP

            self._Vf_kern = MLP(dimX, 1, [128, 128], fixed_batch_size=bsz)

        self.solver = self._solver_maker(max_iter=max_iter, print_time=print_time)

    def _get_meta_constraints(self):
        rmax = 10000.0
        model = self.model
        lb_xk = np.asarray(model.X_merge(-rmax, -rmax, -rmax, 1e-1, -1, -1, -1, -1))
        ub_xk = np.asarray(model.X_merge(rmax, rmax, rmax, 500, 1, 1, 1, 1))
        assert np.all(lb_xk <= ub_xk)
        omega1_max = math.pi / 4
        omega1_min = -omega1_max
        nx_max = 5
        nx_min = -1
        ny_max = 0.2
        ny_min = -ny_max
        nz_max = 1
        nz_min = -8
        lb_uk = np.asarray(model.U_merge(nx_min, ny_min, nz_min, omega1_min))
        ub_uk = np.asarray(model.U_merge(nx_max, ny_max, nz_max, omega1_max))
        assert np.all(lb_uk <= ub_uk)
        return lb_xk, ub_xk, lb_uk, ub_uk

    def _make_constraints(self, var_Ubar, var_Xbar, param_X0, param_Xd, param_ts):
        """
        TODO: register with name
        """
        names = {}
        g = []
        lbg = []
        ubg = []
        use_single_shooting = self.use_single_shooting
        model = self.model
        dt = self.dt
        lb_xk, ub_xk, lb_uk, ub_uk = self._get_meta_constraints()
        self._lb_uk = lb_uk
        self._ub_uk = ub_uk

        idxs_rel = OrderedDict()

        # 初始状态约束
        if use_single_shooting:
            var_Xbar[:, 0] = param_X0
        else:
            Xerr = var_Xbar[:, 0] - param_X0
            for i in range(model.dimX):
                g.append(Xerr[i])
                lbg.append(0)
                ubg.append(0)
                idxs_rel[f"trans_{i}_{0}"] = len(g) - 1
        for k in range(self.horizon):
            tk = param_ts[:, k]
            xk = var_Xbar[:, k]
            uk = var_Ubar[:, k]
            # 状态转移约束
            x2targ = model.ca_fd(tk, xk, uk, dt)
            tnext = tk + dt
            param_ts[:, k + 1] = tnext
            if use_single_shooting:
                var_Xbar[:, k + 1] = x2targ
            else:
                for i in range(model.dimX):
                    g.append(var_Xbar[i, k + 1] - x2targ[i])
                    lbg.append(0)
                    ubg.append(0)
                    idxs_rel[f"trans_{i}_{k+1}"] = len(g) - 1

            # 状态约束
            for i in range(model.dimX):
                g.append(var_Xbar[i, k + 1])
                lbg.append(lb_xk[i])
                ubg.append(ub_xk[i])
                idxs_rel[f"x_{i}_{k+1}"] = len(g) - 1

            # 控制约束
            for i in range(model.dimU):
                g.append(uk[i])
                lbg.append(lb_uk[i])
                ubg.append(ub_uk[i])
                idxs_rel[f"u_{i}_{k}"] = len(g) - 1

        if use_single_shooting:
            self._f_Xp = ca.Function("Xpred", [param_X0, var_Ubar], [var_Xbar])
        g = ca.vertcat(*g)
        lbg = np.asarray(lbg)
        ubg = np.asarray(ubg)
        return g, lbg, ubg

    def _vf_maker(self, use_MX=False):
        _sym = MX.sym if use_MX else SX.sym
        dimX = self.model.dimX
        sym_xf = _sym("xf", dimX)
        sym_xfd = _sym("xfd", dimX)
        ek = sym_xf - sym_xfd
        if self.use_nn_Vf:
            vf = self._Vf_kern(ek)
        else:
            vf = 10 * (ek.T @ (self.Q @ ek))
        Vf = ca.Function("Vf", [sym_xf, sym_xfd], [vf])
        return Vf

    def _make_cost(self, U, X, Xd):
        # fd = self.model.cs_fd
        model = self.model
        horizon = self.horizon
        dimX = model.dimX

        dx = 1
        dz = 0.8 * dx
        self.Qw = np.asarray([1 / dx**2] * 2 + [1 / dz**2] + [1 / 1**2] + [1e-2] * 4)

        uspan = absmax(self._lb_uk, self._ub_uk)
        self.Rw = 1 / uspan**2

        Qk = np.diag(self.Qw)
        Rk = np.diag(self.Rw)
        cost = 0
        for k in range(horizon):
            ek = X[:, k + 1] - Xd[:, k + 1]
            uk = U[:, k]
            cost += ek.T @ Qk @ ek + uk.T @ Rk @ uk

        # 终端代价
        self.Vf = self._vf_maker()
        k = horizon
        vf = self.Vf(X[:, k], Xd[:, k])
        cost += vf
        return cost

    def _solver_maker(self, max_iter: int, print_time: bool = True):

        ts = self.param_ts
        var_Xbar = self.var_Xbar
        var_Ubar = self.var_Ubar
        param_X0 = self.param_X0
        param_Xd = self.param_Xd

        solx = self.solx_merge(var_Ubar, var_Xbar)
        g, lbg, ubg = self._make_constraints(var_Ubar, var_Xbar, param_X0, param_Xd, ts)
        self._lbg = lbg
        self._ubg = ubg

        cost = self._make_cost(var_Ubar, var_Xbar, param_Xd)
        p = self.param_merge(param_X0, param_Xd)
        nlp = {
            "x": solx,
            "f": cost,  # 目标函数
            "g": g,  # 约束条件
            "p": p,  # 初始状态
        }

        solver_opts = {
            "ipopt": {
                "hessian_approximation": "limited-memory",  # 使用拟牛顿法（有限内存近似）
                "max_iter": max_iter,  # 最大迭代次数
                "print_level": 0,  # 打印详细信息
                "warm_start_init_point": "yes",  # 热启动
            },
            "print_time": print_time,
        }
        solver = ca.nlpsol("solver", "ipopt", nlp, solver_opts)  # 编译求解器
        return solver

    def calc_control(self, X0: DM, Xd: DM, prev_sol=None, shift=1):
        solver = self.solver

        p = self.param_merge(X0, Xd)
        opts = dict(p=p, lbg=self._lbg, ubg=self._ubg)
        if prev_sol is not None:
            opts["x0"] = self.sol2x0(prev_sol, shift)  # 热启动
        sol = solver(**opts)
        # self.parse_sol(sol)
        U, X = self.solx_split(sol["x"])
        if self.use_single_shooting:
            X = self._f_Xp(X0, U)  # 预测状态
        u_opt = as_numpy(U[:, 0])
        return u_opt, U, X, sol

    def sol2x0(self, prev_sol, shift: int = 1):
        """计算初始决策变量"""
        solx0 = prev_sol["x"]
        if shift > 0:
            # todo: 移位算子
            U, X = self.solx_split(solx0)
            U[:, :-shift] = U[:, shift:]
            if not self.use_single_shooting:
                X[:, :-shift] = X[:, shift:]
            solx0 = self.solx_merge(U, X)
        return solx0

    def solx_merge(self, U, X=None):
        """merge U and X to a single column vector"""
        if self.use_single_shooting:
            solx = ca.reshape(U, (-1, 1))
        else:
            solx = ca.vertcat(
                ca.reshape(U, (-1, 1)),
                ca.reshape(X, (-1, 1)),
            )
        return solx

    def solx_split(self, solx: DM):
        if self.use_single_shooting:
            U = solx
            X = None
        else:
            szU = self.model.dimU * self.horizon
            U = solx[:szU]
            X = solx[szU:]
            X = X.reshape(self.var_Xbar.shape)
        U = U.reshape(self.var_Ubar.shape)
        return U, X

    def parse_sol(self, sol: dict):
        x_prev = sol["x"]  # 变量值
        cost = sol["f"]
        lam_g = sol["lam_g"]  # 约束乘子（若无约束则为空）
        lam_x = sol["lam_x"]  # 变量边界乘子
        lam_p = sol["lam_p"]  # 初始状态乘子
        x_prev
        pass

    def param_merge(self, X0, Xd):
        X0 = ca.reshape(X0, (-1, 1))
        Xd = ca.reshape(Xd, (-1, 1))
        p = ca.vertcat(X0, Xd)
        return p


def fit_lim(range: Sequence[float | Any], rspan=0.05):
    a = np.min(range)
    b = np.max(range)
    c = (b + a) * 0.5
    r = (b - a) * 0.5 * (1 + rspan)
    if r == 0:
        r = 1e-3
    return c - r, c + r


def absmax(a, b):
    r"""
    $i\mapsto \max(|a_i|,|b_i|)$
    """
    a = np.abs(a)
    b = np.abs(b)
    c = a > b
    y = np.where(c, a, b)
    return y


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
    model = PesudoDOF6Model(batch_size=n)
    simdt = 0.050

    def _f_np(t, x: Sequence, u: Sequence):
        y = model.dynamics(t, x, u)  # Sequence[ shape=(1,n) ]
        return y

    def fd_hand_maker(f_kern):

        def _fd_hand(t, x: Sequence, u: Sequence, dt):
            y = ode_rk45(f_kern, t, x, u, dt)
            return y

        return _fd_hand

    def fd_scipy_maker(f_kern):
        mode = -1
        from scipy.integrate import solve_ivp, odeint, RK45

        def f4scipy(t, x: np.ndarray, u: Sequence[np.ndarray]):
            x = x.reshape(-1, n)
            dx = f_kern(t, _vsplit_keepdim(x), _vsplit_keepdim(u))
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
            y
            return y

        return fd

    x0 = np.random.rand(model.dimX, n)
    u0 = np.random.rand(model.dimU, n)

    fd_sci = fd_scipy_maker(_f_np)
    fd_hand = fd_hand_maker(_f_np)
    fd_ca_MX_ = model._fd_maker(use_MX=True)
    fd_ca_SX_ = model._fd_maker(use_MX=False)

    def fd_ca_MX(t, x: Sequence, u: Sequence, dt) -> Sequence:
        x_ = ca.vcat(x)
        u_ = ca.vcat(u)
        y = fd_ca_MX_(t, x_, u_, dt)
        return ca.horzsplit(y)

    def fd_ca_SX(t, x: Sequence, u: Sequence, dt) -> Sequence:
        x_ = ca.vcat(x)
        u_ = ca.vcat(u)
        y = fd_ca_SX_(t, x_, u_, dt)
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
    nitr


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
    ctrl_interval = 2
    horizon = 5  # 预测时域
    ctrldt = simdt * ctrl_interval
    use_single_shooting = bool(1)
    use_nn = bool(0)
    max_steps = int(240 / ctrldt)
    solver_max_iter = 5  # 求解次数限制
    viz = bool(1)
    print_time = bool(0)

    model = PesudoDOF6Model()
    dimU = model.dimU
    dimX = model.dimX
    policy = MPController(
        model,
        horizon=horizon,
        dt=ctrldt,
        use_single_shooting=use_single_shooting,
        use_nn_vf=use_nn,
        max_iter=solver_max_iter,
        print_time=print_time,
    )

    # 参考轨迹
    pe_ref = np.array([-1000, 500.0, -2000])
    V_ref = 120.0
    qeb_ref = np.array([1.0, 0.0, 0.0, 0.0])
    Xd0 = np.hstack([pe_ref, V_ref, qeb_ref]).reshape(-1, 1)
    Xd_all = np.tile(Xd0, (1, max_steps + horizon))

    x0 = np.hstack([[0.0, 0.0, 0.0], 100 + V_ref, [1.0, 0.0, 0.0, 0.0]])

    Xsim_np = np.zeros((dimX, max_steps + 1))
    Xsim_dm = DM(Xsim_np)
    Xsim_dm[:, 0] = x0  # 初始状态
    ts = np.zeros(max_steps + 1)
    costs = np.zeros_like(ts)

    sim_fd = model.ca_fd  # 状态转移反馈, 设置为与预测模型一致

    if viz:
        np.set_printoptions(precision=4, suppress=True)
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Line3D
        from mpl_toolkits.mplot3d import Axes3D

        plt.ion()
        fig = plt.figure()
        ax1 = cast(Axes3D, fig.add_subplot(1, 2, 1, projection="3d"))
        ax1.set_aspect("equal")
        ax1.invert_zaxis()  # Z 轴反向
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_aspect("auto")

        ax1.scatter(pe_ref[0], pe_ref[1], pe_ref[2], c="cyan", label="ref", marker="*")
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
            ax1.plot(
                [0, 0],
                [0, 0],
                [0, 0],
                label="e" + ("XYZ"[i]),
                color="rgb"[i],
                alpha=0.8,
            )[0]
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

    sol0 = None
    start_time = time.time()
    xk_dm = DM(x0)
    tk_in = time.time() - start_time

    # 预热
    Xd_dm = DM(Xd_all[:, : horizon + 1])
    for k in range(5):
        rst = policy.calc_control(xk_dm, Xd_dm, sol0, shift=0)
        u_opt, U, Xp, sol0 = rst

    # 滚动优化
    dts_infer = []
    dts_step = []
    for k in range(max_steps):
        tk = k * ctrldt

        Xd_dm = DM(Xd_all[:, k : k + horizon + 1])
        rst = policy.calc_control(xk_dm, Xd_dm, sol0)
        u_opt, U, Xp, sol0 = rst

        costk = sol0["f"]
        costk = as_numpy(costk).item()
        V0p = as_numpy(policy.Vf(xk_dm, Xd_dm[:, 0])).item()
        Vfp = as_numpy(policy.Vf(Xp[:, -1], Xd_dm[:, -1])).item()

        ts[k] = tk
        costs[k] = costk
        tk_out = time.time() - start_time
        dts_infer.append(tk_out - tk_in)

        # 状态转移
        X1 = Xsim_dm[:, k]
        X2 = X1
        for _ in range(ctrl_interval):
            X2 = sim_fd(tk, X2, u_opt, simdt)

        Xsim_dm[:, k + 1] = X2
        Xsim_np[:, [k + 1]] = X2
        xk_dm = X2
        tk_real = time.time() - start_time
        dts_step.append(tk_real - tk_in)
        tk_in = tk_real

        print(
            " ".join(
                [
                    f"k:{k}",
                    f"tk:{tk:.3f}s",
                    f"dt_infer:{dts_infer[-1]*1e3:.0f}/{np.mean(dts_infer[-horizon:])*1e3:.0f}",
                    f"dt_step_total:{dts_step[-1]*1e3:.0f}/{np.mean(dts_step[-horizon:])*1e3:.0f}",
                ]
            )
        )

        if viz:
            # 真实轨迹
            k0 = max(0, k - 2 * horizon)
            pos_real = Xsim_np[:3, k0 : k + 2]
            _rend_pos(line_pos, *pos_real)

            rwin = np.max(np.abs(pos_real[:, 0] - pos_real[:, -1]))
            rwin = max(1, rwin)

            # 预测轨迹
            pos_pred = as_numpy(Xp[:3, :])
            _rend_pos(line_pos_pred, *pos_pred)

            # 体轴系
            X2_np = Xsim_np[:, [k + 1]]
            _rst = model.X_split(X2_np)
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
                xs = Xsim_np[0, idxs]
                ys = Xsim_np[1, idxs]
                zs = Xsim_np[2, idxs]
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

            _rst = model.U_split(u_opt)
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
                        "dt_infer: {:.6f}s".format(np.mean(dts_infer[idxs])),
                        "dt_step:  {:.6f}s".format(np.mean(dts_step[idxs])),
                    ]
                )
            )

            # fig.tight_layout()
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.ion()  # 注意顺序, 画图->交互->暂停, 否则会慢
            plt.pause(0.005)
            plt.ioff()

    if viz:
        plt.show(block=True)
