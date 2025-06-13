# 公共头文件
# 在numpy下的反向优化操作:
# 1. 乱用 numbda.jit
# 2. 用 split 而不是 [...]
import math
from typing import Any, Sequence, TypeAlias, TypeVar, Union, overload
import numpy as bkbn
from numpy.typing import NDArray as NDArray

cat = bkbn.concatenate
stack = bkbn.stack
sin = bkbn.sin
cos = bkbn.cos
tan = bkbn.tan
atan2 = bkbn.arctan2
asin = bkbn.arcsin
acos = bkbn.arccos
where = bkbn.where
abs_ = bkbn.abs
sum_ = bkbn.sum
prod = bkbn.prod
pow = bkbn.power
sqrt = bkbn.sqrt
norm = bkbn.linalg.norm
cross = bkbn.cross
clip = bkbn.clip
clamp = clip
hypot = bkbn.hypot
reshape = bkbn.reshape
empty = bkbn.empty
zeros = bkbn.zeros
ones = bkbn.ones
eye = bkbn.eye
zeros_like = bkbn.zeros_like
ones_like = bkbn.ones_like
empty_like = bkbn.empty_like
broadcast_arrays = bkbn.broadcast_arrays
deg2rad = bkbn.deg2rad
rad2deg = bkbn.rad2deg
isclose = bkbn.isclose
chunk = bkbn.array_split

unsqueeze = bkbn.expand_dims
squeeze = bkbn.squeeze
permute = bkbn.transpose
swapaxes = bkbn.swapaxes
asarray = bkbn.asarray
isfinite = bkbn.isfinite
isnan = bkbn.isnan
isinf = bkbn.isinf
land = bkbn.logical_and
lor = bkbn.logical_or
lnot = bkbn.logical_not
lxor = bkbn.logical_xor
eq = bkbn.equal
neq = bkbn.not_equal

PI = math.pi
TAU = math.tau  # 2*pi

ndarray = bkbn.ndarray
FloatNDArr = NDArray[bkbn.float32]
DoubleNDArr = NDArray[bkbn.float64]
BoolNDArr = NDArray[bkbn.bool_]
IntNDArr = NDArray[bkbn.int32]
LongNDArr = NDArray[bkbn.int_]
LongLongNDArr = NDArray[bkbn.int64]
CharNDArr = NDArray[bkbn.str_]
#
_SupportedScalar = Union[int, float, bool, bkbn.number, bkbn.bool_]
NDArrOrNum = Union[ndarray, _SupportedScalar]  # only for annotations

_PI = math.pi
_2PI = math.tau
_NPI = -_PI
_PI_HALF = _PI * 0.5


def unbind_keepdim(x: ndarray, axis: int = -1):  # !!!该实现无法被 jit
    axis = axis % x.ndim
    i1 = [slice(None)] * axis
    return [x[*i1, i : i + 1] for i in range(x.shape[axis])]  # (...,1,...)


def unbind(x: ndarray, axis: int = 0):  # !!!该实现无法被 jit
    """torch.unbind 的 numpy 实现"""
    axis = axis % x.ndim
    i1 = [slice(None)] * axis
    return [x[*i1, i] for i in range(x.shape[axis])]  # (...,1,...)


def split_(x: ndarray, split_size_or_sections: int | Sequence[int], axis: int = -1):
    """torch.split 的 numpy 实现"""
    if isinstance(split_size_or_sections, Sequence):
        idxs = bkbn.cumsum(split_size_or_sections)[:-1]
        rst = bkbn.split(x, idxs, axis=axis)
    else:
        rst = bkbn.split(x, split_size_or_sections, axis=axis)
    return rst


def flatten(x: ndarray, start_dim: int = 0, end_dim: int = -1) -> ndarray:
    """
    torch.flatten 的 numpy 实现
    """
    end_dim = end_dim % x.ndim
    start_dim = start_dim % x.ndim
    assert start_dim <= end_dim
    return x.reshape(x.shape[:start_dim] + (-1,) + x.shape[end_dim + 1 :])


def unflatten(x: ndarray, dim: int, sizes: Sequence[int]) -> ndarray:
    """
    torch.unflatten 的 numpy 实现
    """

    dim = dim % x.ndim
    newshape = x.shape[:dim] + (*sizes,) + x.shape[dim + 1 :]
    return x.reshape(newshape)


def norm_(x, p=2, dim=-1, keepdim=True) -> ndarray:
    r"""
    Args:
        x: Input tensor of shape=(..., d).
    Returns:
        $\|x\|_2$, shape=(...,1)
    """
    return norm(x, p, dim, keepdim)


def affcmb(a, b, t) -> ndarray:
    """Affine combination of two tensors.

    $$
    (1-t)*a+t*b
    $$

    Args:
        a: First scalar or tensor, shape=(...,d|1).
        b: Second scalar or tensor, shape=(...,d|1).
        t: Weights for the affine combination, shape=(..., d|1).

    Returns:
        Affine combination of a and b, shape=(...,d).
    """
    a = asarray(a)
    return a + (b - a) * t


def affcmb_inv(a, b, y) -> ndarray:
    """
    仿射组合的逆运算

    Args:
        a: First scalar or tensor, shape=(...,d|1).
        b: Second scalar or tensor, shape=(...,d|1).
        y: a+w*(b-a), shape=(..., d|1)
    Returns:
        w: 仿射系数, shape=(...,d).
    """
    a = asarray(a)
    m = b - a
    y_ = y - a
    eps = 1e-6
    _mis0 = abs_(m) < eps
    _fix = _mis0 + 0.0
    w = (y_ * (1 - _fix)) / (m + _fix)
    return w


def B01toI(x):
    """B(0,1)=[-1,1]->I=[0,1]"""
    return (x + 1) * 0.5


def ItoB01(x):
    """I=[0,1]->B(0,1)=[-1,1]"""
    return x * 2 - 1


def _normalize(x: ndarray, eps: float = 1e-9) -> ndarray:
    x = x / norm_(x).clip(eps, None)
    return x


def normalize(x: ndarray, eps: float = 1e-6) -> ndarray:
    r"""Normalizes a given input tensor to unit length.

    $$
    x^0:=x/\|x\|_2
    $$

    约定: $x^0=O_d \forall x:\|x\|<=\epsilon$

    Args:
        x: Input tensor of shape=(N, d).
        eps: A small value to avoid division by zero. Defaults to 1e-9.

    Rets:
        Normalized tensor of shape=(N, d).
    """
    return _normalize(x, eps)


def is_normalized(v: ndarray, eps: float = 1e-6) -> BoolNDArr:
    r"""Checks if a given input tensor is normalized.

    Args:
        v (_Tensor): (...,d)
        eps (float, optional): tolerance for ||v|-1|. Defaults to 1e-6.
    Returns:
        _Tensor: (...,1)
    """
    return abs_(norm_(v) - 1) <= eps


def modin(x: NDArrOrNum, a: NDArrOrNum, m: NDArrOrNum) -> ndarray:
    r"""
    a+((x-a) mod m)
    if m=0, return a
    if m>0, y $\in [a,a+m)$
    if m<0, y $\in (a-m,a]$
    Args:
        x: 输入数组, shape=(...,)
        a: 周期开始值, shape=(...,)
        m: 周期长度, shape=(...,)
    """
    x = asarray(x)
    if not (isinstance(a, ndarray) and isinstance(m, ndarray)):
        _0 = zeros([1] * x.ndim, dtype=x.dtype)
        if not isinstance(a, ndarray):
            a = _0 + a  # assert: 广播相容
        if not isinstance(m, ndarray):
            m = _0 + m  # assert: 广播相容
    y = where(m == 0, a, (x - a) % m + a)
    return y


def modrad(x, a: NDArrOrNum = -_PI, m: NDArrOrNum = _2PI):
    return modin(x, a, m)


def moddeg(x, a: NDArrOrNum = -180, m: NDArrOrNum = 360):
    return modin(x, a, m)


def delta_reg(a: NDArrOrNum, b: NDArrOrNum, r: NDArrOrNum = _PI) -> ndarray:
    r"""
    计算 a-b 在 mod R=(-r,r] 上的最小幅度值

    即 $\argmin_{ d\in R=(-r,r]: d=a-b (mod |R|)} |d|$
    Args:
        a: target, scalar|shape=(...,)
        b: input, scalar|shape=(...,)
        r: 周期半径, scalar|shape=(...,)
    """
    a = asarray(a)
    r = asarray(r)  # assert all(r > 0)
    diam = r + r
    d = modin(a - b, 0, diam)  # in [0,2r)
    d = where(d <= r, d, d - diam)  # in (-r,r]
    return d


def delta_deg_reg(a, b):
    r"""$\argmin_{ d\in R=(-180,180]: d=a-b (mod |R|)} |d|$"""
    return delta_reg(a, b, 180)


def delta_rad_reg(a, b):
    r"""$$\argmin_{ d\in R=(-pi,pi]: d=a-b (mod |R|)} |d|$"""
    return delta_reg(a, b, _PI)


def rpy_reg(rpy_rad: ndarray) -> ndarray:
    r"""
    计算 roll pitch yaw 在 $S=[0,2\pi)\times[-\pi/2,\pi/2]\times[0,2\pi)$ 上的最近等价元

    $$
    \argmin_{(r,p,y)\in S: R_3(y')R_2(p')R_1(r')=R_3(y)R_2(p)R_1(r)} \|(r,p,y)-(r',p',y')\|_2
    $$
    Args:
        rpy_rad: roll pitch yaw, shape=(...,3)
    Returns:
        rpy: 等价元, shape=(...,3)
    """
    r = rpy_rad[..., 0:1]
    p = rpy_rad[..., 1:2]
    y = rpy_rad[..., 2:3]
    r = modin(r, 0.0, _2PI)
    p = modin(p, _NPI, _2PI)
    y = modin(y, 0.0, _2PI)
    idxs = abs_(p) > _PI_HALF
    p = where(idxs, modin(_PI - p, _NPI, _2PI), p)
    r = where(idxs, modin(r + _PI, 0.0, _2PI), r)
    y = where(idxs, modin(y + _PI, 0.0, _2PI), y)
    rpy = cat([r, p, y], axis=-1)
    return rpy
