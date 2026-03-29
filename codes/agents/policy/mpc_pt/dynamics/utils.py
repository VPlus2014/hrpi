from __future__ import annotations
from typing import Callable, TypeVar, Union
import numpy as np
import torch

TArr = Union[np.ndarray, torch.Tensor]
T_NDArr_co = TypeVar("T_NDArr_co", bound=TArr, covariant=True)

BatchDynamicsType = Callable[[T_NDArr_co, T_NDArr_co], T_NDArr_co]
BatchODESolverType = Callable[
    [BatchDynamicsType, T_NDArr_co, T_NDArr_co, float | T_NDArr_co], T_NDArr_co
]


def ode_euler(f: BatchDynamicsType | Callable, x, u, dt):
    y = x + dt * f(x, u)
    return y


def ode_rk23(f: BatchDynamicsType | Callable, x, u, dt):
    h2 = dt * 0.5
    k1 = f(x, u)
    k2 = f(x + h2 * k1, u)
    y = x + dt * k2
    return y


def ode_rk45(f: BatchDynamicsType | Callable, x, u, dt):
    h2 = dt * 0.5
    k1 = f(x, u)
    k2 = f(x + h2 * k1, u)
    k3 = f(x + h2 * k2, u)
    k4 = f(x + dt * k3, u)
    y = x + (dt / 6.0) * (k1 + 2.0 * (k2 + k3) + k4)
    return y


def shape_rjust(x: T_NDArr_co, target: TArr) -> T_NDArr_co:
    """padding left with 1 to match target shape"""
    if x.ndim < target.ndim:
        shp = (1,) * (int(target.ndim) - int(x.ndim)) + x.shape
        return x.reshape(shp)  # type: ignore
    else:
        return x
