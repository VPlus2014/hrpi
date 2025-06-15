from __future__ import annotations
from typing import cast
from ._head import *

# from numba import jit


def _calc_zem1(
    los: ndarray,
    dlos: ndarray,
    tmin: float | ndarray = 0.0,
    tmax: float | ndarray = math.inf,
    eps: float = 1e-12,
) -> tuple[ndarray, ndarray]:
    assert len(los.shape) == len(dlos.shape), (
        "p1,v1,p2,v2 must have the same dims, got.",
        len(los.shape),
        len(dlos.shape),
    )
    dp = los
    dv = dlos
    pv: ndarray = (dp * dv).sum(-1, keepdims=True)  # (...,1)
    vv: ndarray = (dv * dv).sum(-1, keepdims=True)  # (...,1)
    # v\dot (p+t v)=0
    t_miss = -pv / vv.clip(eps, None)  # \in\R; |dv|=0 -> t_*=0
    t_miss = clip(t_miss, tmin, tmax)  # 投影
    ibad = ~isfinite(t_miss)
    if ibad.any():
        raise ValueError("t_miss contains NaN or inf.", where(ibad))
    md = norm(dp + dv * t_miss, axis=-1, keepdims=True)  # (...,1)
    return md, t_miss


def calc_zem1(
    los: ndarray,
    dlos: ndarray,
    tmin: float | ndarray = 0.0,
    tmax: float | ndarray = math.inf,
    eps: float = 1e-12,
) -> tuple[ndarray, ndarray]:
    r"""
    零控脱靶量(1阶)

    $$
    \min_\{d(t)=\|p+t*v\|_2 | t\in T=[t_\min,t_\max]\}
    $$

    Args:
        los: shape=(...,d) 视线
        dlos: shape=(...,d) 视线速度
        tmin: 最小时间, float|shape=(...,1), 默认为 0
        tmax: 最大时间, float|shape=(...,1), 默认为 inf
    Returns:
        MD: 脱靶量 shape=(...,1)\
                $MD :=\min_{t\in T} d(t)$\

        t_miss: 脱靶时间 shape=(...,1)\
                $t_* :=\min{t\in T: d(t)=MD}$\
    """
    return _calc_zem1(los, dlos, tmin, tmax, eps)


def _calc_zem2(
    los: ndarray,
    dlos: ndarray,
    ddlos: ndarray,
    tmin: float | ndarray = 0.0,
    tmax: float | ndarray = math.inf,
):
    raise NotImplementedError


# !!!
def _los_is_visible_v1(
    los: ndarray,  # (...,N,d)
    rball: ndarray,  # (...,N,1)
    mask: BoolNDArr,  # (...,N,1)
):
    r"""
    $$
    F(i,j):=(I_1 p_i)\cap \bar B(p_j,r_j)=\emptyset\\
    visible_i:= m_i\land \bigwedge_{j\neq i}(m_j\to F(i,j))\\
    
    G(i,j):= (i\neq j)\land(m_i\land m_j)\land F(i,j)
    mixed_i:= \bigvee_{j\neq i} G(i,j)\land G(j,i)
    $$

    """
    n = los.shape[-2]
    _0f3 = zeros_like(los)
    #
    mi = unsqueeze(mask, -2)  # (...,N,1,1)
    mj = unsqueeze(mask, -3)  # (...,1,N,1)
    dij = calc_zem1(-los, los, 0.0, 1.0)[0]  # (...,N,N,1) |p_j-\Pi_{I_1 p_i}p_j|
    rj = unsqueeze(rball, -3)  # (...,1,N,1)
    fij = dij > rj  # (...,N,N,1) (I_1 p_i)\cap\bar B(p_j,r_j)=\emptyset
    #
    _I = eye(n, dtype=bkbn.bool_).reshape(
        [1] * len(los.shape[:-2]) + [n, n, 1]
    )  # \neg (i\neq j)
    vis = cast(BoolNDArr, mask & ((_I | (~mj) | fij).all(-2)))  # (...,N,1)
    #
    bij = (~_I) & (mi & mj) & (~fij)  # (...,N,N,1) i blocked by j
    # version 1.0
    # b1i = bij.any(-2)  # (...,N,1)
    # b2i = bij.any(-3)  # (...,N,1)
    # mix = b1i & b2i  # (...,N,1)
    # version 1.1
    bji = bij.transpose(-2, -3)
    mixij = bij & bji
    mix = cast(BoolNDArr, mixij.any(-2))  # (...,N,1)
    return vis, mix


def los_is_visible(
    los: ndarray, rball: ndarray, mask: BoolNDArr | None = None
) -> tuple[BoolNDArr, ...]:
    r"""
    判断视线是否不被任何球遮挡
    Args:
        los (_Tensor): 视线组 shape=(...,N,d)
        rball (_Tensor): 球半径 shape=(...,N,1)
        mask (_Tensor | None, optional): 掩码,确认各位的信息是否可用, shape=(...,N,1);
            Defaults to None->默认全部可用.
    Returns:
        visible (_Tensor): 视线是否不受阻, shape=(...,N,1)
        mixed (_Tensor): 是否混淆(对visible=False的元素用), shape=(...,N,1)
    """
    if mask is None:
        # mask = _ones(len(los.shape) * [1], dtype=bool, device=los.device)
        mask = ones_like(rball, dtype=bool)
    assert len(los.shape) == len(rball.shape) == len(mask.shape) >= 2, (
        "los, radius, and mask must have the same dims>=2, got",
        len(los.shape),
        len(rball.shape),
        len(mask.shape),
    )
    assert rball.shape[-1] == 1, ("rball.shape[-1] must be 1, got", rball.shape[-1])
    assert mask.shape[-1] == 1, ("mask.shape[-1] must be 1, got", mask.shape[-1])
    n = los.shape[-2]
    assert rball.shape[-2] == 1 or n == rball.shape[-2], (
        "rball.shape[-2] must be 1 or",
        n,
        "bug got",
        rball.shape[-2],
    )
    assert mask.shape[-2] == 1 or n == mask.shape[-2], (
        "mask.shape[-2] must be 1 or",
        n,
        "bug got",
        mask.shape[-2],
    )
    return _los_is_visible_v1(los, rball, mask)


def _vec_cosine(v1: ndarray, v2: ndarray, n1: ndarray, n2: ndarray, eps: float = 1e-6):
    # v1_is_zero = n1 < eps
    # v2_is_zero = n2 < eps
    # any_zero = v1_is_zero | v2_is_zero
    # c = where(
    #     any_zero,
    #     zeros([1] * any_zero.ndim, n1.dtype),
    #     sum_(v1 * v2, -1, keepdims=True) / (n1 * n2).clip(eps, None),
    # )
    c = sum_(v1 * v2, -1, keepdims=True) / (n1 * n2).clip(eps, None)
    return c


def vec_cosine(
    v1: ndarray,
    v2: ndarray,
    n1: ndarray | float | None = None,
    n2: ndarray | float | None = None,
):
    r"""
    计算两个向量的余弦相似度
    $$
    \cos\angle(v1,v2)=\frac{v1\cdot v2}{|v1||v2|}
    $$
    Args:
        v1 (_NDArr): 向量1 shape: (...,3)
        v2 (_NDArr): 向量2 shape: (...,3)
        n1 (_NDArr|float|None): 向量1的长度 shape: (...,1) or scalar
        n2 (_NDArr|float|None): 向量2的长度 shape: (...,1) or scalar

    Returns:
        _NDArr: 余弦相似度 shape: (...,1)
    """
    if not isinstance(n1, ndarray):
        n1 = norm_(v1)
    if not isinstance(n2, ndarray):
        n2 = norm_(v2)
    return _vec_cosine(v1, v2, n1, n2)
