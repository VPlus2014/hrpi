# 插值模块
from __future__ import annotations
from ._head import *
from .rotation import *


def _quat_slerp(q_0: ndarray, q_1: ndarray, t: ndarray) -> ndarray:
    q_0 = quat_normalize(q_0)
    q_1 = quat_normalize(q_1)
    dq = quat_mul(q_1, quat_inv(q_0))  # (...,4)

    # 钝角检测, 解决过渡点的双倍覆盖问题
    dq = quat_rect_re(dq)  # ->\alpha\in[0,pi]
    re_dq, im_dq = quat_split2(dq)
    ha = acos(re_dq)
    ax = normalize(im_dq)  # (...,3)

    tha = ha * t  # (...,N)
    tha = unsqueeze(tha, axis=-1)  # (...,N,1)
    costha = cos(tha)  # (...,N,1)
    sintha = sin(tha)  # (...,N,1)
    ax = unsqueeze(ax, -2)  # (...,1,3)

    tdq = cat([costha, sintha * ax], axis=-1)  # (...,N,4)
    q_0 = unsqueeze(q_0, -2)  # (...,1,4)

    q_t = quat_mul(tdq, q_0)  # (...,N,4)
    return q_t


def quat_slerp(q_0: ndarray, q_1: ndarray, t: ndarray) -> ndarray:
    r"""

    规范四元数插值 $t\mapsto (q_1 q_0^{-1})^t q_0, t\in[0,1]$

    Args:
        q_0: shape: (...,4)
        q_1: shape: (...,4)
        t: shape: (...,N), N 为插值的点数
    Returns:
        q_t: shape: (...,N,4), 插值后的规范四元数
    """
    return _quat_slerp(q_0, q_1, t)


def herp2_(
    y0: ndarray,
    dy0: ndarray,
    y1: ndarray,
    dy1: ndarray,
    t: ndarray,
) -> ndarray:
    r"""
    2点3次 Hermite 插值(张量版)

    Args:
        y0: f(0), shape: (...,1|d).
        dy0: \dot f(0), shape: (...,1|d).
        y1: f(1), shape: (...,1|d).
        dy1: \dot f(1), shape: (...,1|d).
        t: 内插时间\in[0,1], shape: (...,1|d).
    Returns:
        yt: shape: (...,d), 插值后的点
    """
    assert y0.ndim == dy0.ndim == y1.ndim == dy1.ndim == t.ndim, (
        "expect all input have same dim, got",
        y0.ndim,
        dy0.ndim,
        y1.ndim,
        dy1.ndim,
        t.ndim,
    )
    # h=1
    alpha = 3 * (dy0 + dy1) + 6 * (y0 - y1)  # (...,d)
    tt = t**2
    tthf = 0.5 * tt
    yt = y0 - (tthf - 0.5) * dy0 + tthf * dy1 + ((t / 3 - 0.5) * tt) * alpha
    return yt

    position = cat([y0, y1], axis=-1)
    velocity = cat([dy0, dy1], axis=-1)
    _1 = zeros_like(position[..., :1])
    _0 = ones_like(_1)
    x = cat(
        [
            _0,
            _1,
        ],
        axis=-1,
    )

    r, c1 = x.shape
    _, c2 = t.shape

    x_prime_1 = x.repeat(1, c1).reshape(r, c1, c1)
    x_prime_1 = permute(x_prime_1, (0, 2, 1))
    x_prime_mask = eye(c1, dtype=x.dtype).repeat(r, 1, 1)
    x_prime_2 = x_prime_1 - x_prime_mask * x_prime_1
    x_prime = unsqueeze(x_prime_2, 1).repeat(1, c2, 1, 1)

    t_prime_1 = unsqueeze(t, 2).repeat(1, 1, c1)
    t_prime = unsqueeze(t_prime_1, 3).repeat(1, 1, 1, c1)

    l_num_1 = t_prime - x_prime
    l_num_mask = eye(c1, device=device).repeat(r * c2, 1, 1).reshape(r, c2, c1, c1)
    l_num_2 = l_num_1 - l_num_mask * l_num_1 + l_num_mask
    l_num = prod(l_num_2, dim=2)

    l_den_1 = prod((x_prime_1.permute(0, 2, 1) - x_prime_1) + x_prime_mask, dim=1)
    l_den = unsqueeze(l_den_1, 1).repeat(1, c2, 1)

    l = l_num / l_den

    l_prime_1 = 1.0 / ((x_prime_1.permute(0, 2, 1) - x_prime_1) + x_prime_mask)
    l_prime_2 = sum_(l_prime_1, dim=1) - 1
    l_prime = unsqueeze(l_prime_2, 1).repeat(1, c2, 1)

    # Create x_prime_3 for functions A and B
    x_prime_3 = unsqueeze(x, 1).repeat(1, c2, 1)

    # Calculate function B
    B = (t_prime_1 - x_prime_3) * l * l

    # Calculate function A
    A = (1 - 2 * (t_prime_1 - x_prime_3) * l_prime) * l * l

    # Calculate final result H
    A_prime = unsqueeze(position, 1).repeat(1, c2, 1) * A
    B_prime = unsqueeze(velocity, 1).repeat(1, c2, 1) * B
    H = sum_(A_prime, dim=2) + sum_(B_prime, dim=2)
    return H


def herp2(
    y0: ndarray,
    dy0: ndarray,
    y1: ndarray,
    dy1: ndarray,
    t: ndarray,
) -> ndarray:
    r"""
    2点3次 Hermite 插值

    Args:
        y0: f(0), shape: (...,1|d).
        dy0: \dot f(0), shape: (...,1|d).
        y1: f(1), shape: (...,1|d).
        dy1: \dot f(1), shape: (...,1|d).
        t: 内插时间\in[0,1], shape: (...,N), N 为插值的点数.
    Returns:
        yt: shape: (...,d), 插值后的点
    """
    assert y0.ndim == dy0.ndim == y1.ndim == dy1.ndim == t.ndim
    y0 = unsqueeze(y0, -2)  # (...,1,d)
    y1 = unsqueeze(y1, -2)  # (...,1,d)
    dy0 = unsqueeze(dy0, -2)  # (...,1,d)
    dy1 = unsqueeze(dy1, -2)  # (...,1,d)
    t = unsqueeze(t, axis=-1)  # (...,N,1)
    return herp2_(y0, dy0, y1, dy1, t)


def lerp(
    a: ndarray,
    b: ndarray,
    t: ndarray,
) -> ndarray:
    r"""
    线性插值 $t\mapsto a+t*(b-a)$

    Args:
        a (_Tensor): (...,d), 端点1
        b (_Tensor): (...,d), 端点2
        t (_Tensor): (...,N), N 为插值的点数

    Returns:
        c: (...,N,d) 插值后的点
    """
    assert len(a.shape) == len(b.shape) == len(t.shape)
    assert a.shape[-1] == b.shape[-1]
    # 别 reapeat 了，自动 broadcast 少1/3时间
    a = unsqueeze(a, -2)  # (...,1,d)
    b = unsqueeze(b, -2)  # (...,1,d)
    t = unsqueeze(t, axis=-1)  # (...,N,1)
    v_t = a + t * (b - a)
    return v_t


def nlerp(a: ndarray, b: ndarray, t: ndarray) -> ndarray:
    r"""
    规范化线性插值 $t\mapsto \frac{1}{\|(1-t)a+t*b\|} [(1-t)a+t*b]$
    Args:
        a (_Tensor): (...,d), 端点1
        b (_Tensor): (...,d), 端点2
        t (_Tensor): (...,N), N 为插值的点数

    Returns:
        c: (...,N,d) 线性插值后的单位向量(线性插值产生的零向量仍是零向量)
    """
    c = lerp(a, b, t)  # (...,N,d)
    c = normalize(c)
    return c
