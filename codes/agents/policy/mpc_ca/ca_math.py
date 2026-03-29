# 兼容 numpy 与 casadi 的矩阵类运算
from __future__ import annotations
from typing import (
    Any,
    Callable,
    List,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
    no_type_check,
    overload,
)
import casadi as ca
import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeAlias

CaMatLike = Union[ca.DM, ca.MX, ca.SX]
ArithElem = Union[float, int, np.number, np.ndarray, CaMatLike]
T_ArithElem_co = TypeVar("T_ArithElem_co", bound=ArithElem, covariant=True)
_T = TypeVar("_T")
Tuple4 = tuple[_T, _T, _T, _T]
Tuple3 = tuple[_T, _T, _T]


def as_numpy(v: CaMatLike | NDArray | Any) -> "NDArray[np.floating]":
    if isinstance(v, np.ndarray):
        return v
    if isinstance(v, (ca.SX, ca.MX)):
        v = ca.DM(v)  # convert SX to DM iff all elements are set

    if isinstance(v, ca.DM):
        v = v.full()
        return v  # type: ignore
    return np.asarray(v)


@overload
def vunbind_keepdim(x: CaMatLike) -> list[CaMatLike]: ...
@overload
def vunbind_keepdim(x: np.ndarray | Sequence) -> list[np.ndarray]: ...


def vunbind_keepdim(
    x: CaMatLike | np.ndarray | Sequence,
) -> list[CaMatLike] | list[np.ndarray]:
    if isinstance(x, CaMatLike):
        return ca.vertsplit(x)
    elif not isinstance(x, np.ndarray):
        x = np.asarray(x)
    # assert x_.ndim == 2, ("x must be 2-dimensional", x_.ndim)
    return np.vsplit(x, len(x))


def cross(
    a: Sequence[T_ArithElem_co], b: Sequence[T_ArithElem_co]
) -> Tuple3[T_ArithElem_co]:
    r"""
    叉乘
    Args:
        a: 3维向量, len=3 or shape=(3,n)
        b: 3维向量, len=3 or shape=(3,n)
    Returns:
        $a\times b$ 分量, len=3
    """
    a1, a2, a3 = a[0], a[1], a[2]
    b1, b2, b3 = b[0], b[1], b[2]
    return (
        a2 * b3 - a3 * b2,
        a3 * b1 - a1 * b3,
        a1 * b2 - a2 * b1,
    )  # type: ignore


def quat_rot(
    q: Sequence[T_ArithElem_co], x: Sequence[T_ArithElem_co]
) -> Tuple3[T_ArithElem_co]:
    """
    四元数旋转 Q*(0,x)*Q^*
    assert len(q) == 4 and len(x) == 3
    assert q is normaized |q|==1
    Args:
        q: 四元数, len=4 or shape=(4,n)
        x: 3维向量, len=3 or shape=(3,n)
    Returns:
        旋转后的3维向量分量, len=3
    """
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    x1, x2, x3 = x[0], x[1], x[2]
    _2q0 = q0 + q0
    _2q01 = _2q0 * q1
    _2q02 = _2q0 * q2
    _2q03 = _2q0 * q3
    cosa = _2q0 * q0 - 1
    _2_qv_x = q1 * x1 + q2 * x2 + q3 * x3
    _2_qv_x = _2_qv_x + _2_qv_x
    # cos_a*x + (1-cos_a)*(qv^0,x)*x + 2*cos_a*(qv^0)\times x
    y1 = cosa * x1 + _2_qv_x * q1 + (_2q02 * x3 - _2q03 * x2)
    y2 = cosa * x2 + _2_qv_x * q2 + (_2q03 * x1 - _2q01 * x3)
    y3 = cosa * x3 + _2_qv_x * q3 + (_2q01 * x2 - _2q02 * x1)
    return y1, y2, y3  # type: ignore


def quat_mul(
    p: Sequence[T_ArithElem_co], q: Sequence[T_ArithElem_co]
) -> Tuple4[T_ArithElem_co]:
    """
    四元数乘法
    Args:
        p: LHS quaternion, len=4 or shape=(4,n)
        x: RHS quaternion, len=4 or shape=(4,n)
    Returns:
        旋转后的3维向量分量, len=3
    """
    p0, p1, p2, p3 = p[0], p[1], p[2], p[3]
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    r0 = p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3
    r1 = p0 * q1 + q0 * p1 + p2 * q3 - p3 * q2
    r2 = p0 * q2 + q0 * p2 + p3 * q1 - p1 * q3
    r3 = p0 * q3 + q0 * p3 + p1 * q2 - p2 * q1
    return r0, r1, r2, r3  # type: ignore


def quat_conj(q: Sequence[T_ArithElem_co]) -> Tuple4[T_ArithElem_co]:
    """
    四元数共轭
    """
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    return q0, -q1, -q2, -q3  # type: ignore


def sumsqr(x: Sequence[T_ArithElem_co]) -> T_ArithElem_co:
    ss = None
    for i in x:
        x2 = i * i
        if ss is None:
            ss = x2
        else:
            ss = ss + x2
    return ss  # type: ignore


def norm_2(x: Sequence[T_ArithElem_co]) -> T_ArithElem_co:
    """
    2-norm
    Args:
        x: vectors, len=d or shape=(d,n)
    Returns:
        $||x||_2$
    """
    ss = sumsqr(x)
    if isinstance(ss, CaMatLike):
        return ca.sqrt(ss)
    else:
        return np.sqrt(ss)  # type: ignore


def normalize_2(
    x: Sequence[T_ArithElem_co], eps: float | None = None
) -> tuple[T_ArithElem_co, ...]:
    """
    2-norm normalization
    Args:
        x: vectors, len=d or shape=(d,n)
        eps: small value to avoid division by zero, eps<=0 no protection
    Returns:
        normalized vectors, len=d or shape=(d,n)
    """
    xn = norm_2(x)
    if eps is not None and eps > 0:
        bad = xn <= eps
        if isinstance(xn, CaMatLike):
            xn = ca.if_else(bad, 1, xn)
        else:
            xn = np.where(bad, 1, xn)
    y = [el / xn for el in x]
    return tuple(y)  # type: ignore


quat_normalize = normalize_2
quat_norm = norm_2

DynamicFuncType: TypeAlias = Callable[
    [
        Sequence[T_ArithElem_co],  # $0: X
        Sequence[T_ArithElem_co],  # $1: U
    ],
    Sequence[T_ArithElem_co],  # return $0: \dot{X}
]
ODESolverType: TypeAlias = Callable[
    [
        DynamicFuncType,  # $0: f
        Sequence[T_ArithElem_co],  # $1: X_t
        Sequence[T_ArithElem_co],  # $2: U
        T_ArithElem_co,  # $3: dt
    ],
    Sequence[T_ArithElem_co],  # return $0: X_{t+dt}
]


def ode_euler(
    f: DynamicFuncType,
    x: Sequence[T_ArithElem_co],
    u: Sequence[T_ArithElem_co],
    dt: T_ArithElem_co,
):
    if isinstance(x, CaMatLike):
        d = x.shape[0]
    else:
        d = len(x)
    k1 = f(x, u)
    y = [x[i] + dt * k1[i] for i in range(d)]  # type: ignore
    return y


def ode_rk23(
    f: DynamicFuncType,
    x: Sequence[T_ArithElem_co],
    u: Sequence[T_ArithElem_co],
    dt: T_ArithElem_co,
):
    """modified euler method"""
    if isinstance(x, CaMatLike):
        d = x.shape[0]
    else:
        d = len(x)
    h2 = dt * 0.5
    k1 = f(x, u)
    k2 = f([x[i] + h2 * k1[i] for i in range(d)], u)  # type: ignore
    y = [x[i] + dt * k2[i] for i in range(d)]  # type: ignore
    return y


def ode_rk45(
    f: DynamicFuncType,
    x: Sequence[T_ArithElem_co],
    u: Sequence[T_ArithElem_co],
    dt: T_ArithElem_co,
):
    """Dormand-Prince method"""
    if isinstance(x, CaMatLike):
        d = x.shape[0]
    else:
        d = len(x)
    h2 = dt * 0.5
    k1 = f(x, u)
    k2 = f([x[i] + h2 * k1[i] for i in range(d)], u)  # type: ignore
    k3 = f([x[i] + h2 * k2[i] for i in range(d)], u)  # type: ignore
    k4 = f([x[i] + dt * k3[i] for i in range(d)], u)  # type: ignore
    h_ = dt / 6
    y = [x[i] + h_ * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(d)]  # type: ignore
    return y


T_MatLike_co = TypeVar(
    "T_MatLike_co", bound=Union[CaMatLike, np.ndarray], covariant=True
)


def traj_single_shooting(
    fd: Callable[[T_MatLike_co, T_MatLike_co], T_MatLike_co] | ca.Function | Callable,
    x0: T_MatLike_co,  # (dimX,B)
    ubar: T_MatLike_co,  # (dimU,horizon*B)
    horizon: int,
    batch_size: int,
    out_xbar: T_MatLike_co,
):
    out_xbar[:, :batch_size] = x0
    for k in range(horizon):
        i1 = slice(k * batch_size, (k + 1) * batch_size)
        i2 = slice(i1.start + batch_size, i1.stop + batch_size)
        x1: T_MatLike_co = out_xbar[:, i1]  # type: ignore
        u1: T_MatLike_co = ubar[:, i1]  # type: ignore
        out_xbar[:, i2] = fd(x1, u1)
    return out_xbar
