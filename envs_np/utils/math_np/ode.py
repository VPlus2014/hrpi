# 简化版ODE求解器
from __future__ import annotations
from typing import Callable, Sequence, Union

from ._head import *

_DynamicsFuncType = Union[
    Callable[
        [
            NDArrOrNum,  # $0 时间 t
            ndarray,  # $1 状态组 X
        ],
        ndarray,  # returns 导数 dotX
    ],
    Callable[
        [
            Any,  # $0 时间 t
            ndarray,  # $1 状态组 X
        ],
        ndarray,  # returns 导数 dotX
    ],
    Callable,
]


def ode_euler(
    f: _DynamicsFuncType,
    t0: NDArrOrNum,
    x0: ndarray,
    dt: NDArrOrNum,
) -> ndarray:
    """欧拉法"""
    k1 = f(t0, x0)
    # x_next = [xi + dt * wi for xi, wi in zip(x0, w1)]
    x_next = x0 + dt * k1
    return x_next


def ode_rk23(
    f: _DynamicsFuncType,
    t0: NDArrOrNum,
    x0: ndarray,
    dt: NDArrOrNum,
) -> ndarray:
    """改进欧拉法(中点法)"""
    h2 = dt * 0.5
    k1 = f(t0, x0)
    k2 = f(t0 + h2, x0 + h2 * k1)
    x_next = x0 + dt * k2
    return x_next


def ode_rk45(
    f: _DynamicsFuncType,
    t0: NDArrOrNum,
    x0: ndarray,
    dt: NDArrOrNum,
) -> ndarray:
    """定步长 4阶 Runge-Kutta 法"""
    h2 = dt * 0.5
    tmid = t0 + h2

    k1 = f(t0, x0)
    k2 = f(tmid, x0 + h2 * k1)
    k3 = f(tmid, x0 + h2 * k2)
    k4 = f(t0 + dt, x0 + dt * k3)
    x_next = x0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_next
