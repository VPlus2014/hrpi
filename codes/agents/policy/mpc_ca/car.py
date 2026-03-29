import math
from typing import Callable, List
import casadi as _ca
from casadi import SX
from matplotlib.pylab import norm
import numpy as np
import matplotlib.pyplot as _plt
from controllers.casadi_ext import *


def fit_lim(range: List[float], rspan=0.05):
    a = np.min(range)
    b = np.max(range)
    c = (b + a) * 0.5
    r = (b - a) * 0.5 * (1 + rspan)
    if r == 0:
        r = 1e-3
    return c - r, c + r


g = 9.81


class Car:
    # MPC 汽车模型
    def __init__(self, dt: float = 1e-3, ctrl_interval: int = 10):
        self.dimX = 5
        self.dimU = 2
        self.dt = dt
        self.ctrl_interval = ctrl_interval
        self.L = 1.0  # 车长

    @classmethod
    def X_split(cls, x):
        p = x[:2]
        V = x[2]
        theta = x[3]  # 车轮夹角
        psi = x[4]  # 车身偏航角
        return p, V, theta, psi

    @classmethod
    def X_merge(cls, p, V, theta, psi):
        x = _ca.vcat([p, V, theta, psi])
        return x

    def X_rectify(self, x):
        p, V, theta, psi = self.X_split(x)
        theta = ca.atan2(ca.sin(theta), ca.cos(theta))
        psi = ca.atan2(ca.sin(psi), ca.cos(psi))
        x = self.X_merge(p, V, theta, psi)
        return x

    def is_term(self, x_err):
        # 终止条件
        p_err, V_err, theta_err, psi_err = self.X_split(x_err)
        p_done = ca.norm_1(p_err) < 0.1
        V_done = ca.fabs(V_err) < 0.1
        # theta_done = ca.fabs(theta_err) < 0.01
        # psi_done = ca.fabs(psi_err) < 0.01
        term = ca.logic_and(p_done, V_done)
        # term = ca.logic_and(term, theta_done)
        # term = ca.logic_and(term, psi_done)
        return term

    def f(self, x, u):
        # 线性化汽车动力学模型
        L = self.L
        xy, V, theta, psi = self.X_split(x)

        dot_xy = ca.vcat([V * ca.cos(psi), V * ca.sin(psi)])
        rho = 0.011
        dot_V = u[0] * g - (rho * 0.5) * (ca.sign(V) * V**2)
        dot_theta = u[1]
        dot_psi = V * ca.tan(theta) / L
        dotX = self.X_merge(dot_xy, dot_V, dot_theta, dot_psi)
        return dotX

    def fd(self, x, u):
        dt = self.dt
        sim_acc = self.ctrl_interval
        for k in range(sim_acc):
            y1 = x + self.f(x, u) * dt
            y2 = x + self.f(y1, u) * dt
            x = 0.5 * (y1 + y2)
            #
            x = self.X_rectify(x)
        return x


def calc_span(a, b):
    a = np.abs(a)
    b = np.abs(b)
    c = a > b
    y = np.where(c, a, b)
    return y


def main():
    simdt = 1e-3
    ctrl_interval = 10
    ctrldt = simdt * ctrl_interval
    sim = Car(dt=simdt, ctrl_interval=ctrl_interval)
    # 初始状态
    x0 = np.hstack([[0.0, 0.0], 334.0, [math.radians(0), math.radians(0)]])
    dimX = x0.shape[0]
    dimU = sim.dimU

    #
    nx_max = 10
    nx_min = -10
    dtheta_max = math.pi
    dtheta_min = -dtheta_max
    theta_max = math.radians(15)
    U_max = np.array([nx_max, dtheta_max])
    U_min = np.array([nx_min, dtheta_min])
    U_span = calc_span(U_max, U_min)

    # 预测时域
    N = 10
    max_iter = 50
    Nshow = 100

    # 参考轨迹
    pe_ref = np.array([0, -100.0])
    V_ref = 0.1
    theta_ref = math.radians(0)
    psi_ref = math.radians(0)

    X_ref = sim.X_merge(pe_ref, V_ref, theta_ref, psi_ref)

    # 权重矩阵
    use_term_only = False
    wQ = np.asarray([1 / 0.1] * 2 + [1 / 10] + [1 / 100] * 2) ** 2 / N
    wR = 1 / U_span**2 * 1e-2

    Q = np.diag(wQ)  # 状态权重矩阵
    R = np.diag(wR)  # 控制输入权重矩阵

    # 定义优化变量
    U = _ca.SX.sym("U", dimU, N)  # 控制输入序列
    X = _ca.SX.sym("X", dimX, N + 1)  # 状态序列

    # 目标函数
    assert N > 0, "预测步数必须大于0"
    gamma = 0.1 ** (1 / N)
    gammas = gamma ** np.arange(N)
    COST = 0
    for k in range(N):
        uk = U[:, k]
        X1 = X[:, k]
        X2 = sim.fd(X1, uk)
        X[:, k + 1] = X2

        u = U[:, k]
        x_err = X2 - X_ref
        nonterm = 1 - sim.is_term(x_err)

        isfinal = k + 1 == N  # 达到预测边界
        if isfinal:
            # pe, V, qeb = dof6_X_split(X2)
            # tgo = ca.norm_2(pe - pe_ref) / (ca.fmax(V, 1e-2) * ctrldt)
            tgo = 1 / (1 - gamma)
            Qk = Q * tgo
            Rk = R * tgo
        else:
            Qk = Q
            Rk = R
        Lk = 1e-3 + _ca.mtimes([u.T, Rk, u])  # 能量代价
        if not use_term_only or isfinal:
            Lk += _ca.mtimes([x_err.T, Qk, x_err])

        gk = gammas[k]
        COST += gk * (nonterm * Lk)

    # 约束条件
    gs = []
    glb = []
    gub = []
    # 动态约束
    for k in range(N):
        xk = X[:, k]
        p, V, theta, psi = sim.X_split(xk)
        gs.append(theta)
        glb.append(-theta_max)
        gub.append(theta_max)
        for i in range(dimU):
            gs.append(U[i, k])
            glb.append(U_min[i])
            gub.append(U_max[i])

    # 将约束条件组合成向量
    gs = _ca.vertcat(*gs)

    # 定义优化问题
    nlp = {
        "x": _ca.reshape(U, (-1, 1)),
        "f": COST,  # 目标函数
        "g": gs,  # 约束条件
        "p": X[:, 0],  # 初始状态
    }
    opts = {
        "ipopt": {
            "hessian_approximation": "limited-memory",  # 使用拟牛顿法（有限内存近似）
            "max_iter": max_iter,  # 最大迭代次数
            "print_level": 5,  # 打印详细信息
        },
        "print_time": True,
    }

    # 创建求解器
    solver = _ca.nlpsol("solver", "ipopt", nlp, opts)

    # 模拟参数
    T = int(60 / (simdt * ctrl_interval))  # 总时间步数
    x_sim = np.zeros((dimX, T + 1))
    x0 = sim.X_rectify(x0)  # 初始状态
    x_sim[:, 0] = ca_to_numpy(x0).ravel()  # 初始状态
    ts = []
    costs = []

    def sim_fd_maker() -> Callable[[np.ndarray, np.ndarray], DM]:
        x = _ca.SX.sym("x", dimX)
        u = _ca.SX.sym("u", dimU)
        fd = ca.Function("fd", [x, u], [sim.fd(x, u)])
        return fd

    # 滚动时域控制循环
    sim_fd = sim_fd_maker()

    from mpl_toolkits.mplot3d.art3d import Line3D
    from mpl_toolkits.mplot3d import Axes3D

    fig = _plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_aspect("equal")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_aspect("auto")

    ax1.scatter(pe_ref[0], pe_ref[1], c="g", label="ref", marker="*")
    line1 = ax1.plot([], [], label="pos")[0]
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.legend()

    line2 = ax2.plot([], [], label="cost")[0]
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Cost")

    _plt.ion()
    prev_us = None
    for k in range(T):
        sol_opts = dict(p=x_sim[:, k], lbg=glb, ubg=gub)
        if prev_us is not None:  # 热启动
            sol_opts["x0"] = prev_us.reshape((-1, 1))

        # 求解优化问题
        sol = solver(**sol_opts)

        us_opt = sol["x"]
        us_opt = _ca.reshape(us_opt, U.shape)
        us_opt = ca_to_numpy(us_opt)
        u_opt = us_opt[:, 0]
        #
        us_1_N = us_opt[:, 1:]
        us_N = us_opt[:, [-1]]
        prev_us = np.concat([us_1_N, us_N], axis=-1)

        cost = sol["f"]
        cost = ca_to_numpy(cost).item()
        tk = k * ctrldt
        ts.append(tk)
        costs.append(cost)

        # 状态更新
        X1 = x_sim[:, k]
        X2 = sim_fd(X1, u_opt)
        # x2 = dof6_fd(x1, u_opt, dt=dt)
        X2 = ca_to_numpy(X2).ravel()

        # 应用控制输入，更新状态
        x_sim[:, k + 1] = X2

        pe, V, theta, psi = sim.X_split(X2)
        X_err = X2 - X_ref
        done = ca_to_numpy(sim.is_term(X_err)).item()

        los = pe_ref - pe
        losR = norm(los)

        _plt.ioff()
        k2 = k + 2
        k1 = max(0, k2 - Nshow)
        idxs = slice(k1, k2)

        xs = x_sim[0, idxs]
        ys = x_sim[1, idxs]
        line1.set_data(xs, ys)
        xmin = min(np.min(xs), pe_ref[0])
        xmax = max(np.max(xs), pe_ref[0])
        ymin = min(np.min(ys), pe_ref[1])
        ymax = max(np.max(ys), pe_ref[1])
        ax1.set_xlim(fit_lim([xmin, xmax]))
        ax1.set_ylim(fit_lim([ymin, ymax]))
        ax1.set_title("\n".join([f"Time: {tk:.2f} s", f"|LOS|={losR:.2f}, V={V:.2f}"]))
        # ax.relim(True)
        # ax.autoscale(tight=True)

        line2.set_data(ts, costs)
        i1 = max(0, len(ts) - Nshow)
        ax2.set_xlim(*fit_lim([ts[i1], ts[-1]]))
        costs_ = np.asarray(costs[i1:])
        ax2.set_ylim(*fit_lim(costs_))
        ax2.set_title(f"Cost: {cost:.2}")

        fig.tight_layout()
        fig.canvas.draw()
        _plt.pause(0.010)
        _plt.ion()

        if done:
            print("Done!")
            break

    _plt.show(block=True)


if __name__ == "__main__":
    main()
